"""BC pre-training from agent_v1 expert + PPO fine-tuning with pure game reward.

Phase 1: BC — collect expert factory decisions from agent_v1, supervised pre-training.
Phase 2: PPO — fine-tune with pure game reward (r0/100), vs v3 opponent.
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS,
    scout_action, worker_action, miner_action,
    agent as expert_agent,
)

# Load v3 as PPO opponent
import importlib.util
_v3_spec = importlib.util.spec_from_file_location(
    "agent_submit_v3",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_submit_v3.py"),
)
_v3_mod = importlib.util.module_from_spec(_v3_spec)
_v3_spec.loader.exec_module(_v3_mod)
v3_agent = _v3_mod.agent
v3_STATE = _v3_mod.STATE

# ─── Constants ───────────────────────────────────────────────────────

GRID_R = 2
WALL_CH = 5
NUM_SCALARS = 12
INPUT_SIZE = (2*GRID_R+1)**2 * WALL_CH + NUM_SCALARS

ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH",
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER", "IDLE",
]
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


# ─── Feature Extraction ──────────────────────────────────────────────

def extract(obs, config, my_player, occupied):
    factory = None
    for uid, d in obs.robots.items():
        if d[4] == my_player and d[0] == TYPE_FACTORY:
            factory = (uid, d)
            break
    if factory is None:
        return None, None

    uid, data = factory
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    w = config.width
    turn = STATE["turn"]

    grid = np.zeros((5, 5, 5), dtype=np.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nc, nr = c + dc, r + dr
            idx = (nr - obs.southBound) * w + nc
            if (0 <= nc < w and obs.southBound <= nr <= obs.northBound
                    and 0 <= idx < len(obs.walls)):
                v = obs.walls[idx]
                if v != -1:
                    grid[dr+2, dc+2] = [
                        float(bool(v & 1)), float(bool(v & 2)),
                        float(bool(v & 4)), float(bool(v & 8)), 1.0,
                    ]
    wall_flat = grid.flatten()

    sc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_SCOUT)
    wc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_WORKER)
    mc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_MINER)
    has_nodes = float(bool(getattr(obs, "miningNodes", {})))
    stuck = STATE.get("factory_stuck", 0)

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
            if energy >= getattr(config, "workerCost", 200): mask[5] = 1.0
            if energy >= getattr(config, "scoutCost", 50): mask[6] = 1.0
            if has_nodes and energy >= getattr(config, "minerCost", 300): mask[7] = 1.0
    mask[8] = 1.0
    if mask.sum() == 0:
        mask[8] = 1.0

    return features, mask


# ─── Actor-Critic Network ────────────────────────────────────────────

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
        return action, log_prob, value


# ─── Phase 1: Behavioral Cloning ─────────────────────────────────────

def _reset_state():
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}})


def collect_bc_data(num_games=200):
    """Collect (features, action_idx, mask) from agent_v1 expert games."""
    data = []
    print(f"Collecting BC data from {num_games} expert games...")

    for gi in range(num_games):
        _reset_state()
        v3_STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                          "factory_stuck": 0, "walls": {}})
        seed = random.randint(0, 999999)
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)

        def bc_agent(obs, config):
            my_player = obs.player

            # Get expert actions from agent_v1
            expert_actions = expert_agent(obs, config)

            # Re-extract features (agent_v1 already called update_state internally)
            occupied = {}
            for uid2, d2 in obs.robots.items():
                occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

            for uid2, d2 in obs.robots.items():
                if d2[4] == my_player and d2[0] == TYPE_FACTORY:
                    feat, msk = extract(obs, config, my_player, occupied)
                    if feat is not None and uid2 in expert_actions:
                        action_str = expert_actions[uid2]
                        if action_str in ACTION_TO_IDX:
                            ai = ACTION_TO_IDX[action_str]
                            if msk[ai] > 0:
                                data.append((feat.copy(), ai, msk.copy()))
                    break

            return expert_actions

        env.run([bc_agent, v3_agent])
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{num_games} games, {len(data)} samples")

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

            loss = criterion(logits, act_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(act_batch)
            correct += (logits.argmax(dim=-1) == act_batch).sum().item()
            total += len(act_batch)

        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss/total:.4f} | acc={acc:.1f}%")

    return model


# ─── Phase 2: PPO with Pure Game Reward ──────────────────────────────

def run_ppo_game(model, seed, explore=True):
    """Run one game vs v3, collecting factory trajectory."""
    _reset_state()
    v3_STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                      "factory_stuck": 0, "walls": {}})
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    traj = []

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
                feat, msk = extract(obs, config, my_player, occupied)
                if feat is not None:
                    s = torch.FloatTensor(feat).unsqueeze(0)
                    m = torch.FloatTensor(msk).unsqueeze(0)
                    with torch.no_grad():
                        if explore:
                            ai, log_p, val = model.get_action(s, m)
                            ai = ai.item()
                            log_p = log_p.item()
                            val = val.item()
                        else:
                            probs, val = model(s, m)
                            ai = torch.argmax(probs).item()
                            log_p = 0.0
                            val = val.item()
                    traj.append((feat.copy(), ai, msk.copy(), log_p, val))
                    actions[uid2] = ACTIONS[ai]
                break

        return actions

    env.run([ppo_agent, v3_agent])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return traj, r0, r1


def compute_gae(traj, reward, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute GAE from trajectory with pure game reward as terminal signal."""
    T = len(traj)
    if T == 0:
        return [], []

    r = reward / 100.0  # normalized game reward
    values = [traj[i][4] for i in range(T)]

    advantages = [0.0] * T
    returns = [0.0] * T
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
            delta = r - values[t]
        else:
            next_value = values[t + 1]
            delta = gamma * next_value - values[t]  # no per-step reward

        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]

    return advantages, returns


def ppo_update(model, optimizer, all_feat, all_act, all_mask,
               all_old_lp, all_adv, all_ret):
    states = torch.FloatTensor(np.array(all_feat))
    actions = torch.LongTensor(all_act)
    masks = torch.FloatTensor(np.array(all_mask))
    old_log_probs = torch.FloatTensor(all_old_lp)
    advantages = torch.FloatTensor(all_adv)
    returns = torch.FloatTensor(all_ret)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    probs, values = model(states, masks)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((values - returns) ** 2).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()


# ─── Training Loop ───────────────────────────────────────────────────

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
    print(f"=== Train v{version} (BC+PPO, pure game reward) ===")
    print(f"Weights -> {save_path}")

    model = ActorCritic()

    # ── Phase 1: BC Pre-training ──
    bc_data = collect_bc_data(num_games=bc_games)
    if bc_data:
        model = pretrain_bc(model, bc_data, epochs=bc_epochs, lr=lr)
        print("BC pre-training complete.\n")
    else:
        print("WARNING: No BC data, skipping pre-training.\n")

    # ── Phase 2: PPO with pure game reward ──
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_wr = 0
    t0 = time.time()

    for it in range(num_iter):
        all_feat, all_act, all_mask, all_old_lp = [], [], [], []
        all_adv, all_ret = [], []
        wins = 0

        for _ in range(batch):
            seed = random.randint(0, 999999)
            traj, r0, r1 = run_ppo_game(model, seed, explore=True)

            if r0 > r1:
                wins += 1

            advs, rets = compute_gae(traj, r0)
            for i, (feat, ai, msk, lp, val) in enumerate(traj):
                all_feat.append(feat)
                all_act.append(ai)
                all_mask.append(msk)
                all_old_lp.append(lp)
                all_adv.append(advs[i])
                all_ret.append(rets[i])

        if not all_adv:
            continue

        # PPO update epochs
        for _ in range(PPO_EPOCHS):
            pl, vl, el = ppo_update(model, optimizer, all_feat, all_act,
                                     all_mask, all_old_lp, all_adv, all_ret)

        wr = wins / batch * 100
        elapsed = time.time() - t0
        print(f"[{it+1:3d}/{num_iter}] WR={wr:5.1f}% p_loss={pl:.4f} "
              f"v_loss={vl:.4f} ent={el:.3f} n={len(all_feat)} t={elapsed:.0f}s")

        if wr > best_wr:
            best_wr = wr
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best {best_wr:.0f}% saved")

    final_path = f"nn_weights_v{version}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final weights saved to {final_path}")
    return model, version, best_wr


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate_vs_random(model, num_games=500):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        _, r0, r1 = run_ppo_game(model, seed, explore=False)
        if r0 > r1: wins += 1
        elif r0 < r1: losses += 1
        else: draws += 1
    print(f"Eval vs random: {wins}W-{losses}L-{draws}D ({wins/num_games*100:.1f}%)")
    return wins, losses, draws


def export_weights(model, version, path=None):
    if path is None:
        path = f"nn_weights_v{version}.py"
    sd = model.state_dict()
    with open(path, "w") as f:
        f.write('"""Auto-generated NN weights."""\nimport numpy as np\n\nWEIGHTS = {\n')
        for name, tensor in sd.items():
            arr = tensor.detach().numpy()
            f.write(f"    '{name}': np.array({arr.tolist()}, dtype=np.float32),\n")
        f.write('}\n')
    print(f"Weights exported to {path}")


if __name__ == "__main__":
    model, ver, best = train(num_iter=200, batch=50, lr=0.0003,
                              bc_games=200, bc_epochs=20)
    model.load_state_dict(torch.load(f"nn_weights_v{ver}.pt"))
    evaluate_vs_random(model)
    export_weights(model, ver)
