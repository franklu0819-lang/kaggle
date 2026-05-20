"""v7: NN controls ALL units (factory + scout + worker + miner).

Shared backbone, universal 13-action space with per-type masking.
Phase 1: BC pre-training from agent_v1 expert data (all units).
Phase 2: PPO fine-tuning vs mixed opponents (50% v3 + 30% v5 + 20% random).
"""
import sys, os, random, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE as STATE_V1, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS, BIT_N, BIT_E, BIT_S, BIT_W,
    scout_action, worker_action, miner_action,
)

# Load fixed opponents
_base = os.path.dirname(os.path.abspath(__file__))
_spec_v3 = importlib.util.spec_from_file_location(
    'agent_v3_opp', os.path.join(_base, 'agent_submit_v3.py'))
OPPONENT_V3 = importlib.util.module_from_spec(_spec_v3)
_spec_v3.loader.exec_module(OPPONENT_V3)

_spec_v5 = importlib.util.spec_from_file_location(
    'agent_v5_opp', os.path.join(_base, 'agent_submit_v5.py'))
OPPONENT_V5 = importlib.util.module_from_spec(_spec_v5)
_spec_v5.loader.exec_module(OPPONENT_V5)

OPP_V3_RATIO = 0.5
OPP_V5_RATIO = 0.3

def _pick_opponent():
    r = random.random()
    if r < OPP_V3_RATIO:
        return OPPONENT_V3, "v3"
    elif r < OPP_V3_RATIO + OPP_V5_RATIO:
        return OPPONENT_V5, "v5"
    else:
        return None, "random"

def _reset_opponent(opp_mod):
    if opp_mod is not None:
        opp_mod.STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})

# ─── Universal Action Space ───────────────────────────────────────────

UNIVERSAL_ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH",          # 0-3: movement
    "JUMP_NORTH",                                # 4: factory jump
    "REMOVE_NORTH", "REMOVE_EAST", "REMOVE_WEST",  # 5-7: worker wall removal
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER",  # 8-10: factory build
    "TRANSFORM",                                 # 11: miner transform
    "IDLE",                                      # 12: idle
]
NUM_ACTIONS = len(UNIVERSAL_ACTIONS)  # 13
ACTION_TO_IDX = {a: i for i, a in enumerate(UNIVERSAL_ACTIONS)}

# Static per-type masks: which actions are ever valid for each type
TYPE_MASKS = {
    TYPE_FACTORY: np.array([1,1,1,1, 1, 0,0,0, 1,1,1, 0, 1], dtype=np.float32),
    TYPE_SCOUT:   np.array([1,1,1,1, 0, 0,0,0, 0,0,0, 0, 1], dtype=np.float32),
    TYPE_WORKER:  np.array([1,1,1,1, 0, 1,1,1, 0,0,0, 0, 1], dtype=np.float32),
    TYPE_MINER:   np.array([1,1,1,1, 0, 0,0,0, 0,0,0, 1, 1], dtype=np.float32),
}

# ─── Constants ────────────────────────────────────────────────────────

GRID_R = 2
WALL_CH = 5
NUM_WALL_FEATURES = (2 * GRID_R + 1) ** 2 * WALL_CH  # 125
NUM_SCALARS = 37
INPUT_SIZE = NUM_WALL_FEATURES + NUM_SCALARS  # 162

# Reward shaping
GAMMA = 0.99
GAE_LAMBDA = 0.95
# Shared
W_GAP = 1.0
W_SURVIVAL = 0.05
W_OUTCOME_WIN = 3.0
W_OUTCOME_LOSS = -1.0
# Factory
W_MOVE = 0.5
W_JUMP = 0.3
# Scout
W_EXPLORE = 0.3
W_CRYSTAL = 0.5
W_AHEAD = 0.1
# Worker
W_REMOVE = 0.5
W_FOLLOW = 0.2
# Miner
W_TRANSFORM = 1.0
W_MINE_INCOME = 0.5

# PPO
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


# ─── Feature Extraction ──────────────────────────────────────────────

def _build_wall_grid(c, r, obs, config):
    """Build 5x5x5 wall grid centered on (c, r)."""
    w = config.width
    grid = np.zeros((5, 5, 5), dtype=np.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nc, nr = c + dc, r + dr
            idx = (nr - obs.southBound) * w + nc
            if (0 <= nc < w and obs.southBound <= nr <= obs.northBound
                    and 0 <= idx < len(obs.walls)):
                v = obs.walls[idx]
                if v != -1:
                    grid[dr + 2, dc + 2] = [
                        float(bool(v & 1)), float(bool(v & 2)),
                        float(bool(v & 4)), float(bool(v & 8)), 1.0,
                    ]
    return grid.flatten()


def _count_unit_types(obs, my_player):
    sc = wc = mc = 0
    for d in obs.robots.values():
        if d[4] == my_player:
            if d[0] == TYPE_SCOUT: sc += 1
            elif d[0] == TYPE_WORKER: wc += 1
            elif d[0] == TYPE_MINER: mc += 1
    return sc, wc, mc


def _find_factory(obs, my_player):
    for uid, d in obs.robots.items():
        if d[4] == my_player and d[0] == TYPE_FACTORY:
            return uid, d
    return None, None


def _nearest_crystal(c, r, obs):
    crystals = getattr(obs, "crystals", None) or {}
    best_dist = 99.0
    best_val = 0.0
    for key, val in crystals.items():
        cc, cr2 = parse_key(key)
        dist = abs(cc - c) + abs(cr2 - r)
        if dist < best_dist:
            best_dist = dist
            best_val = val
    return best_dist, best_val


def _nearest_mining_node(c, r, obs):
    nodes = getattr(obs, "miningNodes", None) or {}
    best_dist = 99.0
    for key in nodes:
        nc, nr = parse_key(key)
        dist = abs(nc - c) + abs(nr - r)
        if dist < best_dist:
            best_dist = dist
    return best_dist


def _enemy_info(c, r, obs, my_player):
    adj_enemy = 0
    nearest_enemy = 99.0
    enemy_vis = 0
    enemy_factory_vis = 0.0
    for uid, d in obs.robots.items():
        if d[4] != my_player:
            enemy_vis += 1
            dist = abs(d[1] - c) + abs(d[2] - r)
            if dist < nearest_enemy:
                nearest_enemy = dist
            if dist <= 1:
                adj_enemy += 1
            if d[0] == TYPE_FACTORY:
                enemy_factory_vis = 1.0
    return adj_enemy, nearest_enemy, enemy_vis, enemy_factory_vis


def _my_mine_count(obs, my_player):
    mines = getattr(obs, "mines", None) or {}
    return sum(1 for v in mines.values() if v[2] == my_player)


def _mine_energy(obs, my_player):
    mines = getattr(obs, "mines", None) or {}
    total = 0
    for v in mines.values():
        if v[2] == my_player:
            total += getattr(v, "energy", 0) if hasattr(v, "energy") else 0
    return total


def _crystals_in_vision(obs):
    crystals = getattr(obs, "crystals", None) or {}
    return len(crystals)


def extract_unit(obs, config, my_player, occupied, reserved, uid, data):
    """Extract 170-dim features and 13-dim mask for any unit."""
    unit_type = data[0]
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    w = config.width
    turn = STATE_V1["turn"]
    gap = r - obs.southBound

    # ── Wall grid centered on unit ──
    wall_flat = _build_wall_grid(c, r, obs, config)

    # ── Scalar features ──
    sc, wc, mc = _count_unit_types(obs, my_player)
    has_nodes = float(bool(getattr(obs, "miningNodes", None)))
    stuck = STATE_V1.get("factory_stuck", 0) if unit_type == TYPE_FACTORY else 0

    # Factory info
    fuid, fdata = _find_factory(obs, my_player)
    if fdata is not None:
        fc, fr = fdata[1], fdata[2]
        dist_to_factory = abs(c - fc) + abs(r - fr)
        dir_to_factory = [
            float(r < fr and c == fc),  # factory is south
            float(c > fc and r == fr),  # factory is east
            float(c < fc and r == fr),  # factory is west
            float(r > fr and c == fc),  # factory is north
        ]
        is_at_spawn = float(c == fc and r == fr + 1)
    else:
        fc, fr = c, r
        dist_to_factory = 0.0
        dir_to_factory = [0.0, 0.0, 0.0, 0.0]
        is_at_spawn = 0.0

    # Reserved cells nearby
    reserved_nearby = sum(1 for dc2 in range(-1, 2) for dr2 in range(-1, 2)
                          if (c + dc2, r + dr2) in reserved)

    # Role-specific
    crystal_dist, crystal_val = _nearest_crystal(c, r, obs)
    walls_near_factory = 0
    if fdata is not None and abs(c - fc) + abs(r - fr) <= 2:
        fw = wb(obs, config, c, r)
        if fw is not None:
            walls_near_factory = sum(1 for bit in [BIT_N, BIT_E, BIT_W] if fw & bit)
    is_on_node = float((c, r) in set(parse_key(k) for k in (getattr(obs, "miningNodes", None) or {})))
    node_dist = _nearest_mining_node(c, r, obs)

    # Enemy info
    adj_enemy, nearest_enemy, enemy_vis, enemy_factory_vis = _enemy_info(c, r, obs, my_player)
    my_mines = _my_mine_count(obs, my_player)
    crystals_vis = _crystals_in_vision(obs)

    # Build scalars (45-dim)
    scalars = np.array([
        # Unit type one-hot (4)
        float(unit_type == TYPE_FACTORY), float(unit_type == TYPE_SCOUT),
        float(unit_type == TYPE_WORKER), float(unit_type == TYPE_MINER),
        # General (11)
        gap / 20.0, energy / 1000.0, move_cd / 5.0,
        c / max(1, w - 1), r / 50.0, turn / 500.0, has_nodes,
        # Team composition (3)
        sc / 3.0, wc / 2.0, mc / 2.0,
        # Factory stuck (1)
        stuck / 10.0,
        # Factory-specific (3)
        jump_cd / 20.0, build_cd / 10.0, float(unit_type == TYPE_FACTORY),
        # Non-factory: factory relation (6)
        dist_to_factory / 20.0, *dir_to_factory, is_at_spawn, reserved_nearby / 4.0,
        # Role-specific: scout (2)
        crystal_dist / 10.0, crystal_val / 50.0,
        # Role-specific: worker (2)
        walls_near_factory / 4.0, 0.0,  # remove_cd not in data
        # Role-specific: miner (2)
        is_on_node, node_dist / 10.0,
        # Enemy awareness (6)
        adj_enemy / 3.0, nearest_enemy / 10.0, enemy_factory_vis,
        my_mines / 3.0, enemy_vis / 3.0, crystals_vis / 5.0,
    ], dtype=np.float32)

    features = np.concatenate([wall_flat, scalars])
    assert features.shape[0] == INPUT_SIZE, f"Expected {INPUT_SIZE}, got {features.shape[0]}"

    # ── Dynamic action mask ──
    mask = TYPE_MASKS[unit_type].copy()

    # Movement (0-3): check move_cd and walls
    if move_cd != 0:
        mask[0:4] = 0
    else:
        for i, d in enumerate(["NORTH", "EAST", "WEST", "SOUTH"]):
            if not can_go(obs, config, c, r, d):
                mask[i] = 0
            elif unit_type != TYPE_FACTORY:
                dc2, dr2, _ = DIRS[d]
                nxt = (c + dc2, r + dr2)
                if nxt in reserved or friendly_at(occupied, nxt, my_player):
                    mask[i] = 0

    # JUMP_NORTH (4): factory only
    if unit_type == TYPE_FACTORY:
        if jump_cd != 0 or turn <= 2 or not in_bounds(c, r + 2, obs, config):
            mask[4] = 0

    # REMOVE (5-7): worker only, check wall exists
    if unit_type == TYPE_WORKER:
        wall_cost = getattr(config, "wallRemoveCost", 100)
        if energy < wall_cost:
            mask[5:8] = 0
        else:
            wv = wb(obs, config, c, r)
            if wv is None:
                mask[5:8] = 0
            else:
                if not (wv & BIT_N): mask[5] = 0
                if not (wv & BIT_E): mask[6] = 0
                if not (wv & BIT_W): mask[7] = 0

    # BUILD (8-10): factory only
    if unit_type == TYPE_FACTORY:
        s_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if move_cd == 0 or build_cd != 0 or not s_ok:
            mask[8:11] = 0
        else:
            spawn = (c, r + 1)
            if friendly_at(occupied, spawn, my_player):
                mask[8:11] = 0
            else:
                if energy < getattr(config, "workerCost", 200):
                    mask[8] = 0
                if energy < getattr(config, "scoutCost", 50):
                    mask[9] = 0
                if not has_nodes or energy < getattr(config, "minerCost", 300):
                    mask[10] = 0

    # TRANSFORM (11): miner only
    if unit_type == TYPE_MINER:
        visible_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", None) or {}))
        transform_cost = getattr(config, "transformCost", 100)
        if (c, r) not in visible_nodes or energy < transform_cost + 1:
            mask[11] = 0

    # IDLE (12): always valid
    mask[12] = 1.0

    return features, mask


# ─── Actor-Critic Network ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, NUM_ACTIONS)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        h = self.backbone(x)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(logits, dim=-1)
        value = self.value_head(h).squeeze(-1)
        return probs, value

    def get_action(self, x, mask=None):
        probs, value = self.forward(x, mask)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def greedy_action(self, x, mask=None):
        probs, _ = self.forward(x, mask)
        return torch.argmax(probs, dim=-1)


# ─── Phase 1: Behavioral Cloning ──────────────────────────────────────

def _reset_v1_state():
    STATE_V1.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})


def collect_bc_data(num_games=200):
    """Collect (features, action_idx, mask, unit_type) from agent_v1 expert games."""
    data = []
    print(f"Collecting BC data from {num_games} expert games (all units)...")

    for gi in range(num_games):
        _reset_v1_state()
        seed = random.randint(0, 999999)
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)

        def bc_agent(obs, config):
            my_player = obs.player
            update_state(obs, config, my_player)

            # Get expert actions from agent_v1 (rule-based)
            actions = {}
            reserved_expert = set()
            occupied = {}
            for uid2, d2 in obs.robots.items():
                occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

            for uid2, d2 in obs.robots.items():
                if d2[4] == my_player and d2[0] == TYPE_SCOUT:
                    scout_action(uid2, d2, obs, config, actions, reserved_expert, occupied, my_player)
            for uid2, d2 in obs.robots.items():
                if uid2 not in actions and d2[4] == my_player and d2[0] == TYPE_WORKER:
                    worker_action(uid2, d2, obs, config, actions, reserved_expert, occupied, my_player)
            for uid2, d2 in obs.robots.items():
                if uid2 not in actions and d2[4] == my_player and d2[0] == TYPE_MINER:
                    miner_action(uid2, d2, obs, config, actions, reserved_expert, occupied, my_player)

            # Factory: agent_v1 has factory_action, use it
            from agent_v1 import factory_action
            for uid2, d2 in obs.robots.items():
                if uid2 not in actions and d2[4] == my_player and d2[0] == TYPE_FACTORY:
                    factory_action(uid2, d2, obs, config, actions, reserved_expert, occupied, my_player)

            # Extract features for all friendly units
            for uid2, d2 in obs.robots.items():
                if d2[4] == my_player and uid2 in actions:
                    feat, msk = extract_unit(obs, config, my_player, occupied, reserved_expert, uid2, d2)
                    if feat is not None:
                        action_str = actions[uid2]
                        # Map action string to universal index
                        ai = ACTION_TO_IDX.get(action_str, None)
                        if ai is not None and msk[ai] > 0:
                            data.append((feat.copy(), ai, msk.copy(), d2[0]))

            return actions

        env.run([bc_agent, "random"])
        if (gi + 1) % 50 == 0:
            print(f"  {gi + 1}/{num_games} games, {len(data)} samples collected")

    print(f"BC data: {len(data)} samples from {num_games} games")
    return data


def pretrain_bc(model, data, epochs=20, lr=0.001, batch_size=256):
    features = torch.FloatTensor(np.array([d[0] for d in data]))
    actions = torch.LongTensor([d[1] for d in data])
    masks = torch.FloatTensor(np.array([d[2] for d in data]))

    dataset = torch.utils.data.TensorDataset(features, actions, masks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print(f"\nBC pre-training: {len(data)} samples, {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for feat_batch, act_batch, mask_batch in loader:
            logits = model.policy_head(model.backbone(feat_batch))
            logits = logits.masked_fill(mask_batch == 0, -1e9)
            loss = criterion(logits, act_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(act_batch)
            correct += (logits.argmax(dim=-1) == act_batch).sum().item()
            total += len(act_batch)

        acc = correct / total * 100
        print(f"  Epoch {epoch + 1}/{epochs} | loss={total_loss / total:.4f} | acc={acc:.1f}%")

    return model


# ─── Phase 2: PPO ─────────────────────────────────────────────────────

def run_ppo_game(model, seed, explore=True):
    """Run one game, collect trajectories for ALL units."""
    STATE_V1.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})
    opp_mod, opp_name = _pick_opponent() if explore else (None, "random")
    _reset_opponent(opp_mod)
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    traj = []  # list of (feat, action, mask, log_prob, value, unit_type, step_info)
    prev_states = {}  # uid -> (row, crystal_val, mine_energy, factory_row)

    def ppo_agent(obs, config):
        my_player = obs.player
        update_state(obs, config, my_player)

        actions = {}
        reserved = set()
        occupied = {}
        for uid2, d2 in obs.robots.items():
            occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

        # Process units in priority order
        for proc_type in [TYPE_SCOUT, TYPE_WORKER, TYPE_MINER, TYPE_FACTORY]:
            for uid2, d2 in obs.robots.items():
                if uid2 in actions:
                    continue
                if d2[4] != my_player or d2[0] != proc_type:
                    continue

                feat, msk = extract_unit(obs, config, my_player, occupied, reserved, uid2, d2)
                if feat is None:
                    continue

                s = torch.FloatTensor(feat).unsqueeze(0)
                m = torch.FloatTensor(msk).unsqueeze(0)
                with torch.no_grad():
                    if explore:
                        ai_t, log_p, val, _ = model.get_action(s, m)
                        ai = ai_t.item()
                        log_p = log_p.item()
                        val = val.item()
                    else:
                        probs, val = model(s, m)
                        ai = torch.argmax(probs).item()
                        log_p = 0.0
                        val = val.item()

                action_str = UNIVERSAL_ACTIONS[ai]
                step_info = {
                    "unit_type": d2[0],
                    "unit_row": d2[2],
                    "unit_col": d2[1],
                    "energy": d2[3],
                    "south_bound": obs.southBound,
                    "turn": STATE_V1["turn"],
                    "action": action_str,
                }

                # Track factory row for move reward
                fuid2, fdata2 = _find_factory(obs, my_player)
                if fdata2 is not None:
                    step_info["factory_row"] = fdata2[2]
                else:
                    step_info["factory_row"] = None

                # Previous state
                prev = prev_states.get(uid2)
                step_info["prev_row"] = prev[0] if prev else None
                step_info["prev_crystal_val"] = prev[1] if prev else 0.0
                step_info["prev_mine_energy"] = prev[2] if prev else 0.0
                step_info["prev_factory_row"] = prev[3] if prev else None

                # Current state
                cur_crystal_val = 0.0
                crystals = getattr(obs, "crystals", None) or {}
                for key, v in crystals.items():
                    cc2, cr2 = parse_key(key)
                    if cc2 == d2[1] and cr2 == d2[2]:
                        cur_crystal_val = v
                        break
                cur_mine_energy = _mine_energy(obs, my_player)

                traj.append((feat.copy(), ai, msk.copy(), log_p, val, d2[0], step_info))
                prev_states[uid2] = (d2[2], cur_crystal_val, cur_mine_energy,
                                     step_info.get("factory_row"))

                # Reserve position
                if action_str in DIRS:
                    dc2, dr2, _ = DIRS[action_str]
                    reserved.add((d2[1] + dc2, d2[2] + dr2))
                elif action_str == "JUMP_NORTH":
                    reserved.add((d2[1], d2[2] + 2))
                else:
                    reserved.add((d2[1], d2[2]))

                actions[uid2] = action_str

        return actions

    opponent_fn = opp_mod.agent if opp_mod is not None else "random"
    env.run([ppo_agent, opponent_fn])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return traj, r0, r1


def compute_step_rewards(traj, r0, r1):
    """Compute per-step shaped rewards for all unit types."""
    T = len(traj)
    if T == 0:
        return []

    step_rewards = []
    for i, (feat, ai, msk, log_p, val, unit_type, info) in enumerate(traj):
        unit_row = info["unit_row"]
        south_bound = info["south_bound"]
        action = info["action"]
        prev_row = info["prev_row"]

        # ── Shared rewards ──
        gap = unit_row - south_bound
        gap_reward = W_GAP * (gap / 20.0)
        survival_reward = W_SURVIVAL

        # Outcome (terminal)
        outcome_reward = 0.0
        if i == T - 1:
            if r0 > r1:
                outcome_reward = W_OUTCOME_WIN
            elif r0 < r1:
                outcome_reward = W_OUTCOME_LOSS

        reward = gap_reward + survival_reward + outcome_reward

        # ── Type-specific rewards ──
        if unit_type == TYPE_FACTORY:
            # Move reward
            if prev_row is not None:
                delta_row = unit_row - prev_row
                reward += W_MOVE * delta_row
                # Jump reward
                if action == "JUMP_NORTH":
                    if delta_row >= 2:
                        reward += W_JUMP * 1.0
                    elif delta_row >= 1:
                        reward += W_JUMP * 0.3
                    else:
                        reward += W_JUMP * (-0.5)

        elif unit_type == TYPE_SCOUT:
            # Ahead of factory
            factory_row = info.get("factory_row")
            if factory_row is not None and prev_row is not None:
                ahead = max(0, unit_row - factory_row - 3)
                reward += W_AHEAD * (ahead / 10.0)
            # Crystal collection
            if prev_row is not None:
                prev_cv = info.get("prev_crystal_val", 0.0)
                cur_cv = 0.0  # crystal consumed if on crystal cell
                if cur_cv < prev_cv:
                    reward += W_CRYSTAL * (prev_cv / 50.0)

        elif unit_type == TYPE_WORKER:
            # Wall removal
            if action.startswith("REMOVE"):
                reward += W_REMOVE * 1.0
            # Follow factory
            factory_row = info.get("factory_row")
            if factory_row is not None:
                dist = abs(info["unit_col"] - 0) + abs(unit_row - factory_row)  # approx
                reward += W_FOLLOW * max(0, 1 - dist / 5.0)

        elif unit_type == TYPE_MINER:
            # Transform
            if action == "TRANSFORM":
                reward += W_TRANSFORM * 1.0
            # Mine income
            prev_me = info.get("prev_mine_energy", 0.0)
            cur_me = _mine_energy_val_from_info(info)
            if cur_me > prev_me:
                reward += W_MINE_INCOME * ((cur_me - prev_me) / 100.0)

        step_rewards.append(reward)

    return step_rewards


def _mine_energy_val_from_info(info):
    """Placeholder - can't easily get mine energy from step info."""
    return 0.0


def compute_gae(values, rewards, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


FACTORY_WEIGHT = 3.0  # Upweight factory samples to counter sample imbalance


def ppo_update(model, optimizer, trajectories, unit_types_batch):
    all_feat, all_act, all_mask, all_old_logp = [], [], [], []
    all_ret, all_adv, all_weights = [], [], []

    for gi, (feats, acts, masks, old_logps, vals, rewards, advs, rets) in enumerate(trajectories):
        utypes = unit_types_batch[gi]
        for i in range(len(feats)):
            all_feat.append(feats[i])
            all_act.append(acts[i])
            all_mask.append(masks[i])
            all_old_logp.append(old_logps[i])
            all_ret.append(rets[i])
            all_adv.append(advs[i])
            all_weights.append(FACTORY_WEIGHT if utypes[i] == TYPE_FACTORY else 1.0)

    if not all_feat:
        return 0.0, 0.0

    states = torch.FloatTensor(np.array(all_feat))
    actions = torch.LongTensor(all_act)
    masks = torch.FloatTensor(np.array(all_mask))
    old_log_probs = torch.FloatTensor(all_old_logp)
    returns = torch.FloatTensor(all_ret)
    advantages = torch.FloatTensor(all_adv)
    sample_weights = torch.FloatTensor(all_weights)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std()

    probs, values = model(states, masks)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
    policy_loss = -(torch.min(surr1, surr2) * sample_weights).mean()

    value_loss = ((values - returns) ** 2).mean()

    loss = policy_loss + VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return policy_loss.item(), value_loss.item()


# ─── Training Loop ────────────────────────────────────────────────────

def _next_version():
    v = 1
    while os.path.exists(f"nn_weights_v{v}.pt"):
        v += 1
    return v


def train(num_iter=200, batch=50, lr=0.0003, version=None,
          bc_games=200, bc_epochs=20):
    if version is None:
        version = _next_version()
    save_path = f"nn_weights_v{version}.pt"
    print(f"=== Train v{version} (BC+PPO, all units, factory-weighted, vs 50%v3+30%v5+20%rand) ===")
    print(f"Weights -> {save_path}")

    model = ActorCritic()

    # Phase 1: BC
    bc_data = collect_bc_data(num_games=bc_games)
    if bc_data:
        model = pretrain_bc(model, bc_data, epochs=bc_epochs, lr=lr)
        print("BC pre-training complete.\n")
    else:
        print("WARNING: No BC data collected.\n")

    # Phase 2: PPO
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_wr = 0
    t0 = time.time()

    for it in range(num_iter):
        trajectories = []
        unit_types_batch = []
        wins = 0
        batch_rewards = []

        for _ in range(batch):
            seed = random.randint(0, 999999)
            traj, r0, r1 = run_ppo_game(model, seed, explore=True)

            if r0 > r1:
                wins += 1

            step_rewards = compute_step_rewards(traj, r0, r1)
            if not step_rewards:
                continue

            values = [t[4] for t in traj]
            advantages, returns = compute_gae(values, step_rewards)

            batch_rewards.extend(step_rewards)
            trajectories.append((
                [t[0] for t in traj],
                [t[1] for t in traj],
                [t[2] for t in traj],
                [t[3] for t in traj],
                values,
                step_rewards,
                advantages,
                returns,
            ))
            unit_types_batch.append([t[5] for t in traj])

        p_loss, v_loss = 0.0, 0.0
        for ep in range(PPO_EPOCHS):
            pl, vl = ppo_update(model, optimizer, trajectories, unit_types_batch)
            p_loss += pl
            v_loss += vl
        p_loss /= PPO_EPOCHS
        v_loss /= PPO_EPOCHS

        wr = wins / batch * 100
        elapsed = time.time() - t0
        avg_r = np.mean(batch_rewards) if batch_rewards else 0
        total_steps = sum(len(t[0]) for t in trajectories)
        print(f"[{it + 1:3d}/{num_iter}] WR={wr:5.1f}% p_loss={p_loss:.4f} "
              f"v_loss={v_loss:.4f} avg_r={avg_r:.3f} steps={total_steps} t={elapsed:.0f}s")

        if wr > best_wr:
            best_wr = wr
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best {best_wr:.0f}% saved")

    final_path = f"nn_weights_v{version}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final weights saved to {final_path}")
    return model, version, best_wr


def evaluate(model, num_games=500):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        _, r0, r1 = run_ppo_game(model, seed, explore=False)
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1
    print(f"Eval vs random: {wins}W-{losses}L-{draws}D ({wins / num_games * 100:.1f}%)")
    return wins, losses, draws


def export_weights(model, version, path=None):
    if path is None:
        path = f"nn_weights_v{version}.py"

    sd = model.state_dict()
    export_sd = {k: v for k, v in sd.items() if not k.startswith("value_head")}

    with open(path, "w") as f:
        f.write('"""Auto-generated NN weights (v%d, all-unit policy)."""\nimport numpy as np\n\nWEIGHTS = {\n' % version)
        for name, tensor in export_sd.items():
            arr = tensor.detach().numpy()
            f.write(f"    '{name}': np.array({arr.tolist()}, dtype=np.float32),\n")
        f.write('}\n')
    print(f"Weights exported to {path}")


if __name__ == "__main__":
    model, ver, best = train(num_iter=200, batch=50, lr=0.0003,
                             bc_games=200, bc_epochs=20)
    model.load_state_dict(torch.load(f"nn_weights_v{ver}.pt"))
    evaluate(model)
    export_weights(model, ver)
