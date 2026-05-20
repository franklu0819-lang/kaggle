"""Compare v8 (BC+PPO) vs v3 agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS, scout_action, worker_action, miner_action,
)

# ─── Load v8 NN weights ───
ACTIONS = ["NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH",
           "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER", "IDLE"]

INPUT_SIZE = 137
NUM_ACTIONS = 9

import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x, mask=None):
        h = self.backbone(x)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return torch.softmax(logits, dim=-1)

GRID_R = 2
WALL_CH = 5
NUM_SCALARS = 12

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
                    grid[dr+2, dc+2] = [float(bool(v & 1)), float(bool(v & 2)),
                                        float(bool(v & 4)), float(bool(v & 8)), 1.0]
    wall_flat = grid.flatten()
    sc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_SCOUT)
    wc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_WORKER)
    mc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_MINER)
    has_nodes = float(bool(getattr(obs, "miningNodes", {})))
    stuck = STATE.get("factory_stuck", 0)
    scalars = np.array([gap/20.0, energy/1000.0, move_cd/5.0, jump_cd/20.0,
                        build_cd/10.0, c/max(1,w-1), turn/500.0,
                        sc/3.0, wc/2.0, mc/2.0, has_nodes, stuck/10.0], dtype=np.float32)
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
    if mask.sum() == 0: mask[8] = 1.0
    return features, mask

model = ActorCritic()
sd = torch.load("nn_weights_v1.pt")
model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("value_head")})
model.eval()

def v8_agent(obs, config):
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
                    probs = model(s, m)
                ai = torch.argmax(probs).item()
                actions[uid2] = ACTIONS[ai]
            break
    return actions

# Load v3 opponent
import importlib.util
spec = importlib.util.spec_from_file_location("agent_submit_v3", "/Users/leo/Projects/kaggle/maze-crawler/agent_submit_v3.py")
v3_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3_mod)
v3_agent_fn = v3_mod.agent

NUM_GAMES = 500
wins, losses, draws = 0, 0, 0
for i in range(NUM_GAMES):
    seed = i * 137 + 42
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0, "walls": {}})
    v3_mod.STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0, "walls": {}})
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    env.run([v8_agent, v3_agent_fn])
    r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
    if r0 > r1: wins += 1
    elif r0 < r1: losses += 1
    else: draws += 1
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{NUM_GAMES}: {wins}W-{losses}L-{draws}D ({wins/(i+1)*100:.1f}%)")

print(f"\nv8 vs v3: {wins}W-{losses}L-{draws}D ({wins/NUM_GAMES*100:.1f}%)")
