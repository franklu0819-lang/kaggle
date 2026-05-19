"""NN-powered agent: trained policy network for factory decisions.
Non-factory units use rule-based logic from agent_v1.py.
"""
from collections import deque
import numpy as np

from nn_weights import WEIGHTS

STATE = {
    "turn": 0,
    "nodes": set(),
    "last_factory_pos": None,
    "factory_stuck": 0,
}

TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER = 0, 1, 2, 3
BIT_N, BIT_E, BIT_S, BIT_W = 1, 2, 4, 8

DIRS = {
    "NORTH": (0, 1, BIT_N),
    "EAST":  (1, 0, BIT_E),
    "SOUTH": (0, -1, BIT_S),
    "WEST":  (-1, 0, BIT_W),
}
OPPOSITE_BIT = {"NORTH": BIT_S, "EAST": BIT_W, "SOUTH": BIT_N, "WEST": BIT_E}

ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH",
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER", "IDLE",
]
NUM_ACTIONS = len(ACTIONS)

# ─── Utility functions (from agent.py) ────────────────────────────────

def parse_key(key):
    c, r = key.split(",")
    return int(c), int(r)

def in_bounds(c, r, obs, config):
    return 0 <= c < config.width and obs.southBound <= r <= obs.northBound

def wb(obs, config, c, r):
    idx = (r - obs.southBound) * config.width + c
    if 0 <= idx < len(obs.walls):
        w = obs.walls[idx]
        if w != -1:
            return w
    return None

def can_go(obs, config, c, r, d):
    dc, dr, bit = DIRS[d]
    nc, nr = c + dc, r + dr
    if not in_bounds(nc, nr, obs, config):
        return False
    w = wb(obs, config, c, r)
    if w is not None and (w & bit):
        return False
    w2 = wb(obs, config, nc, nr)
    if w2 is not None and (w2 & OPPOSITE_BIT[d]):
        return False
    return True

def update_state(obs, config, my_player):
    STATE["turn"] += 1
    for key in getattr(obs, "miningNodes", {}) or {}:
        STATE["nodes"].add(parse_key(key))
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            pos = (data[1], data[2])
            if STATE["last_factory_pos"] is not None:
                if pos == STATE["last_factory_pos"]:
                    STATE["factory_stuck"] += 1
                else:
                    STATE["factory_stuck"] = 0
            STATE["last_factory_pos"] = pos
            break

def bfs_first_step(start, goals, obs, config, passable_fn, max_nodes=300):
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return None
    q = deque([(start, None)])
    visited = {start}
    best_fd, best_dist = None, 999999
    while q:
        cur, first_d = q.popleft()
        dist = min(abs(cur[0] - g[0]) + abs(cur[1] - g[1]) for g in goals)
        if dist < best_dist:
            best_dist = dist
            best_fd = first_d
        for d in ("NORTH", "EAST", "WEST", "SOUTH"):
            if not passable_fn(obs, config, cur[0], cur[1], d):
                continue
            dc, dr, _ = DIRS[d]
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt in visited:
                continue
            visited.add(nxt)
            fd = first_d or d
            if nxt in goal_set:
                return fd
            q.append((nxt, fd))
            if len(visited) >= max_nodes:
                return best_fd
    return best_fd

def bfs_to_row(start, row, obs, config, passable_fn):
    goals = [(c, row) for c in range(config.width) if in_bounds(c, row, obs, config)]
    return bfs_first_step(start, goals, obs, config, passable_fn)

def friendly_at(occupied, cell, my_player):
    return any(o[1][4] == my_player for o in occupied.get(cell, []))

def try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
    dc, dr, _ = DIRS[d]
    nxt = (c + dc, r + dr)
    if nxt in reserved or friendly_at(occupied, nxt, my_player):
        return False
    actions[uid] = d
    reserved.add(nxt)
    return True

def move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row=None):
    if target_row is None:
        target_row = r + 1
    target_row = min(obs.northBound, target_row)
    step = bfs_to_row((c, r), target_row, obs, config, can_go)
    if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
        return True
    width = config.width
    center = width // 4
    ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]
    for d in ["NORTH"] + ew + ["SOUTH"]:
        if can_go(obs, config, c, r, d):
            if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                return True
    return False

# ─── NN Forward Pass (numpy) ──────────────────────────────────────────

def nn_forward(features, mask):
    x = features
    x = np.maximum(0, x @ WEIGHTS['net.0.weight'].T + WEIGHTS['net.0.bias'])
    x = np.maximum(0, x @ WEIGHTS['net.2.weight'].T + WEIGHTS['net.2.bias'])
    logits = x @ WEIGHTS['net.4.weight'].T + WEIGHTS['net.4.bias']
    logits[mask == 0] = -1e9
    logits -= logits.max()
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()
    return np.argmax(probs)

# ─── Feature Extraction ───────────────────────────────────────────────

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
            if energy >= getattr(config, "workerCost", 200):
                mask[5] = 1.0
            if energy >= getattr(config, "scoutCost", 50):
                mask[6] = 1.0
            if has_nodes and energy >= getattr(config, "minerCost", 300):
                mask[7] = 1.0
    mask[8] = 1.0
    if mask.sum() == 0:
        mask[8] = 1.0

    return features, mask

# ─── Unit Actions (rule-based) ────────────────────────────────────────

def scout_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return
    crystals = [(parse_key(k), v) for k, v in (getattr(obs, "crystals", {}) or {}).items()]
    if crystals:
        best = max(
            [(v / max(1, abs(cell[0] - c) + abs(cell[1] - r)), cell)
             for cell, v in crystals if cell != (c, r)],
            key=lambda x: x[0], default=None,
        )
        if best:
            _, target = best
            step = bfs_first_step((c, r), [target], obs, config, can_go)
            if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                return
    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break
    if factory_pos:
        fc, fr = factory_pos
        target_row = min(obs.northBound, fr + 6)
        half = config.width // 2
        target_col = min(half - 1, c + 3) if c < half else max(half, c - 3)
        step = bfs_first_step((c, r), [(target_col, target_row)], obs, config, can_go)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return
    actions[uid] = "IDLE"
    reserved.add((c, r))

def worker_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    wall_cost = getattr(config, "wallRemoveCost", 100)
    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break
    if factory_pos and (c, r) == (factory_pos[0], factory_pos[1] + 1):
        for d in ("NORTH", "EAST", "WEST"):
            if can_go(obs, config, c, r, d):
                if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return
    if factory_pos and energy >= wall_cost + 20:
        fc, fr = factory_pos
        if c == fc and r == fr + 1:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                actions[uid] = "REMOVE_NORTH"
                reserved.add((c, r))
                return
        if abs(c - fc) + abs(r - fr) <= 2:
            for d, bit in [("NORTH", BIT_N), ("EAST", BIT_E), ("WEST", BIT_W)]:
                w = wb(obs, config, c, r)
                if w is not None and (w & bit):
                    actions[uid] = f"REMOVE_{d}"
                    reserved.add((c, r))
                    return
    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return
    target_row = r + 1
    if factory_pos:
        target_row = min(obs.northBound, factory_pos[1] + 2)
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row):
        return
    actions[uid] = "IDLE"
    reserved.add((c, r))

def miner_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    transform_cost = getattr(config, "transformCost", 100)
    visible_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {}))
    if (c, r) in visible_nodes and energy >= transform_cost + 1:
        actions[uid] = "TRANSFORM"
        reserved.add((c, r))
        return
    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return
    mines = set(parse_key(k) for k in getattr(obs, "mines", {}).keys())
    vis_list = [n for n in visible_nodes if n not in mines]
    rem_list = [n for n in STATE["nodes"] if n not in mines and in_bounds(n[0], n[1], obs, config)]
    all_nodes = vis_list + rem_list
    if all_nodes:
        target = min(all_nodes, key=lambda n: abs(n[0] - c) + abs(n[1] - r))
        step = bfs_first_step((c, r), [target], obs, config, can_go)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return
    actions[uid] = "IDLE"
    reserved.add((c, r))

# ─── Main Agent ───────────────────────────────────────────────────────

def agent(obs, config):
    my_player = obs.player
    update_state(obs, config, my_player)

    actions = {}
    reserved = set()
    occupied = {}
    for uid, data in obs.robots.items():
        cell = (data[1], data[2])
        occupied.setdefault(cell, []).append((uid, data))

    # Non-factory units first (rule-based)
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_SCOUT:
            scout_action(uid, data, obs, config, actions, reserved, occupied, my_player)
    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_WORKER:
            worker_action(uid, data, obs, config, actions, reserved, occupied, my_player)
    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_MINER:
            miner_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    # Factory: neural network
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            feat, msk = extract(obs, config, my_player, occupied)
            if feat is not None:
                ai = nn_forward(feat, msk)
                actions[uid] = ACTIONS[ai]
            break

    return actions
