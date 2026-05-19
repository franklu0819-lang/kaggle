"""Train a neural network policy for factory decisions using REINFORCE.

- PyTorch for training (with autograd)
- Exports weights as numpy for submission agent
- Factory uses NN, other units keep rule-based logic
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
)

# ─── Constants ───────────────────────────────────────────────────────

GRID_R = 2           # 5x5 grid around factory
WALL_CH = 5          # N, E, S, W, known
NUM_SCALARS = 12
INPUT_SIZE = (2*GRID_R+1)**2 * WALL_CH + NUM_SCALARS   # 125 + 12 = 137

ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH",
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER", "IDLE",
]
NUM_ACTIONS = len(ACTIONS)

# Reward shaping
GAMMA = 0.99
W_GAP = 1.0
W_MOVE = 0.5
W_JUMP = 0.3
W_SURVIVAL = 0.1
W_OUTCOME_WIN = 3.0
W_OUTCOME_LOSS = -1.0


# ─── Feature Extraction ──────────────────────────────────────────────

def extract(obs, config, my_player, occupied):
    """Return (features[137], mask[9], factory_row, jump_cd) or (None, None, None, None)."""
    factory = None
    for uid, d in obs.robots.items():
        if d[4] == my_player and d[0] == TYPE_FACTORY:
            factory = (uid, d)
            break
    if factory is None:
        return None, None, None, None  # (features, mask, factory_row, jump_cd)

    uid, data = factory
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    w = config.width
    turn = STATE["turn"]

    # 5x5 wall grid
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

    # Scalar features
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

    # ── Action mask ──
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # Movement
    if move_cd == 0:
        for i, d in enumerate(["NORTH", "EAST", "WEST", "SOUTH"]):
            if can_go(obs, config, c, r, d):
                mask[i] = 1.0

    # Jump
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        mask[4] = 1.0

    # Build
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

    # IDLE always valid
    mask[8] = 1.0
    if mask.sum() == 0:
        mask[8] = 1.0

    return features, mask, r, jump_cd


# ─── Policy Network ──────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x, mask=None):
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return torch.softmax(logits, dim=-1)


# ─── Game Runner ─────────────────────────────────────────────────────

def run_game(policy_net, seed, explore=True):
    """Run one game. Returns (trajectory, r0, r1, total_steps).
    trajectory: [(features, action_idx, mask, step_info), ...]
    """
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    traj = []
    prev_factory_row = [None]
    prev_jump_cd = [None]

    def nn_agent(obs, config):
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
                        probs = policy_net(s, m).squeeze(0)
                    if explore:
                        ai = torch.distributions.Categorical(probs).sample().item()
                    else:
                        ai = torch.argmax(probs).item()
                    step_info = {
                        "factory_row": factory_row,
                        "south_bound": obs.southBound,
                        "prev_factory_row": prev_factory_row[0],
                        "prev_jump_cd": prev_jump_cd[0],
                        "jump_cd": jump_cd,
                        "turn": STATE["turn"],
                    }
                    traj.append((feat.copy(), ai, msk.copy(), step_info))
                    prev_factory_row[0] = factory_row
                    prev_jump_cd[0] = jump_cd
                    actions[uid2] = ACTIONS[ai]
                break

        return actions

    env.run([nn_agent, "random"])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return traj, r0, r1, len(env.steps)


# ─── Reward Computation ──────────────────────────────────────────────

def compute_rewards(traj, r0, r1):
    """Compute per-step shaped rewards from trajectory metadata."""
    T = len(traj)
    if T == 0:
        return []

    step_rewards = []
    for i, (feat, ai, msk, info) in enumerate(traj):
        factory_row = info["factory_row"]
        south_bound = info["south_bound"]
        prev_row = info["prev_factory_row"]
        jcd = info["jump_cd"]

        # A: Gap reward — core survival signal
        gap = factory_row - south_bound
        gap_reward = W_GAP * (gap / 20.0)

        # B: Northward movement reward
        if prev_row is not None:
            delta_row = factory_row - prev_row
            move_reward = W_MOVE * delta_row
        else:
            delta_row = 0
            move_reward = 0.0

        # C: JUMP effectiveness
        if ACTIONS[ai] == "JUMP_NORTH":
            if delta_row >= 2:
                jump_reward = W_JUMP * 1.0
            elif delta_row >= 1:
                jump_reward = W_JUMP * 0.3
            else:
                jump_reward = W_JUMP * (-0.5)
        else:
            jump_reward = 0.0

        # D: Per-step survival bonus
        survival_reward = W_SURVIVAL

        # E: Outcome bonus (terminal only)
        outcome_reward = 0.0
        if i == T - 1:
            if r0 > r1:
                outcome_reward = W_OUTCOME_WIN
            elif r0 < r1:
                outcome_reward = W_OUTCOME_LOSS

        total = gap_reward + move_reward + jump_reward + survival_reward + outcome_reward
        step_rewards.append(total)

    return step_rewards


def compute_returns(step_rewards, gamma=GAMMA):
    """Compute discounted returns: G_t = r_t + gamma * G_{t+1}."""
    T = len(step_rewards)
    returns = [0.0] * T
    G = 0.0
    for t in reversed(range(T)):
        G = step_rewards[t] + gamma * G
        returns[t] = G
    return returns


# ─── Training Loop ───────────────────────────────────────────────────

def _next_version():
    """Find the next available version number for weight files."""
    v = 1
    while os.path.exists(f"nn_weights_v{v}.pt"):
        v += 1
    return v


def train(num_iter=200, batch=50, lr=0.001, version=None):
    if version is None:
        version = _next_version()
    save_path = f"nn_weights_v{version}.pt"
    print(f"Training v{version}, weights -> {save_path}")

    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    best_wr = 0

    t0 = time.time()
    for it in range(num_iter):
        all_feat, all_act, all_mask, all_ret = [], [], [], []
        wins = 0
        batch_step_rewards = []

        for _ in range(batch):
            seed = random.randint(0, 999999)
            traj, r0, r1, total_steps = run_game(policy_net, seed, explore=True)

            if r0 > r1:
                wins += 1

            step_rewards = compute_rewards(traj, r0, r1)
            returns = compute_returns(step_rewards)
            batch_step_rewards.extend(step_rewards)

            for i, (feat, ai, msk, info) in enumerate(traj):
                all_feat.append(feat)
                all_act.append(ai)
                all_mask.append(msk)
                all_ret.append(returns[i])

        # Update policy
        states = torch.FloatTensor(np.array(all_feat))
        actions = torch.LongTensor(all_act)
        masks = torch.FloatTensor(np.array(all_mask))
        returns = torch.FloatTensor(all_ret)

        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()

        probs = policy_net(states, masks)
        log_probs = torch.distributions.Categorical(probs).log_prob(actions)
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wr = wins / batch * 100
        elapsed = time.time() - t0
        avg_sr = np.mean(batch_step_rewards) if batch_step_rewards else 0
        print(f"[{it+1:3d}/{num_iter}] WR={wr:5.1f}% loss={loss.item():.4f} "
              f"avg_r={avg_sr:.3f} steps={len(all_feat)} t={elapsed:.0f}s")

        if wr > best_wr:
            best_wr = wr
            torch.save(policy_net.state_dict(), save_path)
            print(f"  -> New best {best_wr:.0f}% saved")

    # Always save final weights regardless of best
    final_path = f"nn_weights_v{version}_final.pt"
    torch.save(policy_net.state_dict(), final_path)
    print(f"Final weights saved to {final_path}")
    return policy_net, version, best_wr


def evaluate(policy_net, num_games=500):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        _, r0, r1, _ = run_game(policy_net, seed, explore=False)
        if r0 > r1: wins += 1
        elif r0 < r1: losses += 1
        else: draws += 1
    print(f"Eval: {wins}W-{losses}L-{draws}D ({wins/num_games*100:.1f}%)")
    return wins, losses, draws


def export_weights(policy_net, version, path=None):
    """Export weights as Python/numpy for submission agent."""
    if path is None:
        path = f"nn_weights_v{version}.py"
    sd = policy_net.state_dict()
    with open(path, "w") as f:
        f.write('"""Auto-generated NN weights."""\nimport numpy as np\n\nWEIGHTS = {\n')
        for name, tensor in sd.items():
            arr = tensor.detach().numpy()
            f.write(f"    '{name}': np.array({arr.tolist()}, dtype=np.float32),\n")
        f.write('}\n')
    print(f"Weights exported to {path}")


if __name__ == "__main__":
    net, ver, best = train(num_iter=200, batch=50, lr=0.001)
    net.load_state_dict(torch.load(f"nn_weights_v{ver}.pt"))
    evaluate(net)
    export_weights(net, ver)
