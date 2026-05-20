"""v10: Wider vision (7x7) + bigger network for factory policy.

Based on train_v6 (BC+PPO vs mixed opponents).
Changes: GRID_R=3 (7x7 grid, 245 wall features), network 257→192→96→9.
"""
import sys, os, random, time, copy, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE as STATE_V1, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS,
    scout_action, worker_action, miner_action,
)
from agent_v2 import agent as expert_agent, STATE as STATE_V2

# Load fixed opponents (self-contained, each has its own STATE)
_base = os.path.dirname(os.path.abspath(__file__))
_spec_v3 = importlib.util.spec_from_file_location(
    'agent_v3_sub', os.path.join(_base, 'agent_submit_v3.py'))
OPPONENT_V3 = importlib.util.module_from_spec(_spec_v3)
_spec_v3.loader.exec_module(OPPONENT_V3)

_spec_v5 = importlib.util.spec_from_file_location(
    'agent_v5_sub', os.path.join(_base, 'agent_submit_v5.py'))
OPPONENT_V5 = importlib.util.module_from_spec(_spec_v5)
_spec_v5.loader.exec_module(OPPONENT_V5)

# Opponent mixing ratios
OPP_V3_RATIO = 0.5
OPP_V5_RATIO = 0.3
# remaining 0.2 is random

def _pick_opponent():
    """Randomly select opponent: v3 / v5 / random."""
    r = random.random()
    if r < OPP_V3_RATIO:
        return OPPONENT_V3, "v3"
    elif r < OPP_V3_RATIO + OPP_V5_RATIO:
        return OPPONENT_V5, "v5"
    else:
        return None, "random"

def _reset_opponent_state(opp_mod):
    if opp_mod is not None:
        opp_mod.STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})

# ─── Constants ───────────────────────────────────────────────────────

GRID_R = 3           # 7x7 grid around factory
WALL_CH = 5
NUM_SCALARS = 12
INPUT_SIZE = (2 * GRID_R + 1) ** 2 * WALL_CH + NUM_SCALARS  # 245+12=257

ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH",
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER", "IDLE",
]
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Reward shaping (same as train.py)
GAMMA = 0.99
GAE_LAMBDA = 0.95
W_GAP = 1.0
W_MOVE = 0.5
W_JUMP = 0.3
W_SURVIVAL = 0.1
W_OUTCOME_WIN = 3.0
W_OUTCOME_LOSS = -1.0

# PPO hyperparameters
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


# ─── Feature Extraction (same as train.py) ──────────────────────────

def extract(obs, config, my_player, occupied):
    """Return (features[257], mask[9], factory_row, jump_cd) or Nones."""
    factory = None
    for uid, d in obs.robots.items():
        if d[4] == my_player and d[0] == TYPE_FACTORY:
            factory = (uid, d)
            break
    if factory is None:
        return None, None, None, None

    uid, data = factory
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    w = config.width
    turn = STATE_V1["turn"]

    grid = np.zeros((2 * GRID_R + 1, 2 * GRID_R + 1, 5), dtype=np.float32)
    for dr in range(-GRID_R, GRID_R + 1):
        for dc in range(-GRID_R, GRID_R + 1):
            nc, nr = c + dc, r + dr
            idx = (nr - obs.southBound) * w + nc
            if (0 <= nc < w and obs.southBound <= nr <= obs.northBound
                    and 0 <= idx < len(obs.walls)):
                v = obs.walls[idx]
                if v != -1:
                    grid[dr + GRID_R, dc + GRID_R] = [
                        float(bool(v & 1)), float(bool(v & 2)),
                        float(bool(v & 4)), float(bool(v & 8)), 1.0,
                    ]
    wall_flat = grid.flatten()

    sc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_SCOUT)
    wc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_WORKER)
    mc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_MINER)
    has_nodes = float(bool(getattr(obs, "miningNodes", {})))
    stuck = STATE_V1.get("factory_stuck", 0)

    scalars = np.array([
        gap / 20.0, energy / 1000.0, move_cd / 5.0, jump_cd / 20.0,
        build_cd / 10.0, c / max(1, w - 1), turn / 500.0,
        sc / 3.0, wc / 2.0, mc / 2.0, has_nodes, stuck / 10.0,
    ], dtype=np.float32)

    features = np.concatenate([wall_flat, scalars])

    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if move_cd == 0:
        for i, d in enumerate(["NORTH", "EAST", "WEST", "SOUTH"]):
            if can_go(obs, config, c, r, d):
                mask[i] = 1.0
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        mask[4] = 1.0
    s_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
    if move_cd != 0 and build_cd == 0 and s_ok:
        spawn = (c, r + 1)
        if not friendly_at(occupied, spawn, my_player):
            if energy >= getattr(config, "workerCost", 200):
                mask[5] = 1.0
            if energy >= getattr(config, "scoutCost", 50):
                mask[6] = 1.0
            if has_nodes and energy >= getattr(config, "minerCost", 300):
                mask[7] = 1.0
    mask[8] = 1.0
    if mask.sum() == 0:
        mask[8] = 1.0

    return features, mask, r, jump_cd


# ─── Actor-Critic Network ────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_SIZE, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(96, NUM_ACTIONS)
        self.value_head = nn.Linear(96, 1)

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


# ─── Phase 1: Behavioral Cloning ─────────────────────────────────────

def _reset_v1_state():
    STATE_V1.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})


def _reset_v2_state():
    STATE_V2.update({
        "turn": 0, "walls": {}, "nodes": set(), "mines": {},
        "enemy_factory": None, "my_factory": None, "enemy_seen": {},
        "factory_stuck": 0, "factory_last_pos": None,
    })


def collect_bc_data(num_games=200):
    """Collect (features, action_idx, mask) from agent_v2 expert games."""
    data = []
    print(f"Collecting BC data from {num_games} expert games...")

    for gi in range(num_games):
        _reset_v1_state()
        _reset_v2_state()
        seed = random.randint(0, 999999)
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)

        def bc_agent(obs, config):
            my_player = obs.player

            # Update v1 state for feature extraction
            update_state(obs, config, my_player)

            # Get expert actions from agent_v2
            expert_actions = expert_agent(obs, config)

            # Extract features for factory
            occupied = {}
            for uid2, d2 in obs.robots.items():
                occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

            for uid2, d2 in obs.robots.items():
                if d2[4] == my_player and d2[0] == TYPE_FACTORY:
                    feat, msk, _, _ = extract(obs, config, my_player, occupied)
                    if feat is not None and uid2 in expert_actions:
                        action_str = expert_actions[uid2]
                        if action_str in ACTION_TO_IDX:
                            ai = ACTION_TO_IDX[action_str]
                            # Only keep if action is in valid mask
                            if msk[ai] > 0:
                                data.append((feat.copy(), ai, msk.copy()))
                    break

            return expert_actions

        env.run([bc_agent, "random"])
        if (gi + 1) % 50 == 0:
            print(f"  {gi + 1}/{num_games} games, {len(data)} samples collected")

    print(f"BC data: {len(data)} samples from {num_games} games")
    return data


def pretrain_bc(model, data, epochs=20, lr=0.001, batch_size=256):
    """Supervised pre-training on expert data."""
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

            loss = criterion(logits, act_batch)
            # Weight by mask validity (downweight IDLE-only samples)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(act_batch)
            correct += (logits.argmax(dim=-1) == act_batch).sum().item()
            total += len(act_batch)

        acc = correct / total * 100
        print(f"  Epoch {epoch + 1}/{epochs} | loss={total_loss / total:.4f} | acc={acc:.1f}%")

    return model


# ─── Phase 2: PPO ────────────────────────────────────────────────────

def run_ppo_game(model, seed, explore=True):
    """Run one game against mixed opponents (v3/v5/random)."""
    STATE_V1.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})
    opp_mod, opp_name = _pick_opponent() if explore else (None, "random")
    _reset_opponent_state(opp_mod)
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    traj = []
    prev_factory_row = [None]

    def ppo_agent(obs, config):
        my_player = obs.player
        update_state(obs, config, my_player)

        actions = {}
        reserved = set()
        occupied = {}
        for uid2, d2 in obs.robots.items():
            occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

        for uid2, d2 in obs.robots.items():
            if d2[4] == my_player and d2[0] == TYPE_SCOUT:
                scout_action(uid2, d2, obs, config, actions, reserved, occupied, my_player)
        for uid2, d2 in obs.robots.items():
            if uid2 not in actions and d2[4] == my_player and d2[0] == TYPE_WORKER:
                worker_action(uid2, d2, obs, config, actions, reserved, occupied, my_player)
        for uid2, d2 in obs.robots.items():
            if uid2 not in actions and d2[4] == my_player and d2[0] == TYPE_MINER:
                miner_action(uid2, d2, obs, config, actions, reserved, occupied, my_player)

        for uid2, d2 in obs.robots.items():
            if d2[4] == my_player and d2[0] == TYPE_FACTORY:
                feat, msk, factory_row, jump_cd = extract(obs, config, my_player, occupied)
                if feat is not None:
                    s = torch.FloatTensor(feat).unsqueeze(0)
                    m = torch.FloatTensor(msk).unsqueeze(0)
                    with torch.no_grad():
                        if explore:
                            ai, log_p, val, _ = model.get_action(s, m)
                            ai = ai.item()
                            log_p = log_p.item()
                            val = val.item()
                        else:
                            probs, val = model(s, m)
                            ai = torch.argmax(probs).item()
                            log_p = 0.0
                            val = val.item()

                    step_info = {
                        "factory_row": factory_row,
                        "south_bound": obs.southBound,
                        "prev_factory_row": prev_factory_row[0],
                        "jump_cd": jump_cd,
                        "turn": STATE_V1["turn"],
                    }
                    traj.append((feat.copy(), ai, msk.copy(), log_p, val, step_info))
                    prev_factory_row[0] = factory_row
                    actions[uid2] = ACTIONS[ai]
                break

        return actions

    opponent_fn = opp_mod.agent if opp_mod is not None else "random"
    env.run([ppo_agent, opponent_fn])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return traj, r0, r1


def compute_step_rewards(traj, r0, r1):
    """Same shaped reward as train.py."""
    T = len(traj)
    if T == 0:
        return []

    step_rewards = []
    for i, (feat, ai, msk, log_p, val, info) in enumerate(traj):
        factory_row = info["factory_row"]
        south_bound = info["south_bound"]
        prev_row = info["prev_factory_row"]

        gap = factory_row - south_bound
        gap_reward = W_GAP * (gap / 20.0)

        if prev_row is not None:
            delta_row = factory_row - prev_row
            move_reward = W_MOVE * delta_row
        else:
            delta_row = 0
            move_reward = 0.0

        if ACTIONS[ai] == "JUMP_NORTH":
            if delta_row >= 2:
                jump_reward = W_JUMP * 1.0
            elif delta_row >= 1:
                jump_reward = W_JUMP * 0.3
            else:
                jump_reward = W_JUMP * (-0.5)
        else:
            jump_reward = 0.0

        survival_reward = W_SURVIVAL

        outcome_reward = 0.0
        if i == T - 1:
            if r0 > r1:
                outcome_reward = W_OUTCOME_WIN
            elif r0 < r1:
                outcome_reward = W_OUTCOME_LOSS

        step_rewards.append(gap_reward + move_reward + jump_reward + survival_reward + outcome_reward)

    return step_rewards


def compute_gae(values, rewards, dones=None, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute GAE advantages and returns."""
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def ppo_update(model, optimizer, trajectories, epoch_idx):
    """One PPO update epoch on collected trajectories."""
    all_feat, all_act, all_mask, all_old_logp = [], [], [], []
    all_ret, all_adv = [], []

    for (feats, acts, masks, old_logps, vals, rewards, advs, rets) in trajectories:
        for i in range(len(feats)):
            all_feat.append(feats[i])
            all_act.append(acts[i])
            all_mask.append(masks[i])
            all_old_logp.append(old_logps[i])
            all_ret.append(rets[i])
            all_adv.append(advs[i])

    if not all_feat:
        return 0.0, 0.0

    states = torch.FloatTensor(np.array(all_feat))
    actions = torch.LongTensor(all_act)
    masks = torch.FloatTensor(np.array(all_mask))
    old_log_probs = torch.FloatTensor(all_old_logp)
    returns = torch.FloatTensor(all_ret)
    advantages = torch.FloatTensor(all_adv)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std()

    probs, values = model(states, masks)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # Clipped surrogate
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = ((values - returns) ** 2).mean()

    # Total loss
    loss = policy_loss + VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def _next_version():
    v = 1
    while os.path.exists(f"nn_weights_v{v}.pt"):
        v += 1
    return v


def train(num_iter=200, batch=50, lr=0.0003, version=None,
          bc_games=200, bc_epochs=20):
    """Full training: BC pre-training + PPO against fixed v3 opponent."""
    if version is None:
        version = _next_version()
    save_path = f"nn_weights_v{version}.pt"
    print(f"=== Train v{version} (BC+PPO vs 50%v3+30%v5+20%rand, 7x7 vision, 192/96 net) ===")
    print(f"Weights -> {save_path}")

    model = ActorCritic()

    # ── Phase 1: BC Pre-training ──
    bc_data = collect_bc_data(num_games=bc_games)
    if bc_data:
        model = pretrain_bc(model, bc_data, epochs=bc_epochs, lr=lr)
        print("BC pre-training complete.\n")
    else:
        print("WARNING: No BC data collected, skipping pre-training.\n")

    # ── Phase 2: PPO ──
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_wr = 0
    t0 = time.time()

    for it in range(num_iter):
        trajectories = []
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

            values = [t[4] for t in traj]  # val from trajectory
            advantages, returns = compute_gae(values, step_rewards)

            batch_rewards.extend(step_rewards)
            trajectories.append((
                [t[0] for t in traj],  # features
                [t[1] for t in traj],  # actions
                [t[2] for t in traj],  # masks
                [t[3] for t in traj],  # old_log_probs
                values,                 # values
                step_rewards,
                advantages,
                returns,
            ))

        # PPO update epochs
        p_loss, v_loss = 0.0, 0.0
        for ep in range(PPO_EPOCHS):
            pl, vl = ppo_update(model, optimizer, trajectories, ep)
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
    """Evaluate against fixed v3 opponent."""
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
    print(f"Eval vs v3: {wins}W-{losses}L-{draws}D ({wins / num_games * 100:.1f}%)")
    return wins, losses, draws


def export_weights(model, version, path=None):
    """Export weights as Python/numpy for agent_v3.py submission."""
    if path is None:
        path = f"nn_weights_v{version}.py"

    sd = model.state_dict()
    # Only export policy-relevant weights (backbone + policy_head)
    export_sd = {k: v for k, v in sd.items() if not k.startswith("value_head")}

    with open(path, "w") as f:
        f.write('"""Auto-generated NN weights (BC+PPO vs v3, v%d)."""\nimport numpy as np\n\nWEIGHTS = {\n' % version)
        for name, tensor in export_sd.items():
            arr = tensor.detach().numpy()
            f.write(f"    '{name}': np.array({arr.tolist()}, dtype=np.float32),\n")
        f.write('}\n')
    print(f"Weights exported to {path}")


if __name__ == "__main__":
    model, ver, best = train(num_iter=200, batch=50, lr=0.0003,
                             bc_games=200, bc_epochs=20)
    # Load best weights for evaluation
    model.load_state_dict(torch.load(f"nn_weights_v{ver}.pt"))
    evaluate(model)
    export_weights(model, ver)
