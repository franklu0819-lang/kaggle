"""v5 agent: NN controls ALL units (factory + scout + worker + miner).
Shared backbone, universal 13-action space, numpy-only inference.
"""
from collections import deque
import numpy as np

from nn_weights_v7 import WEIGHTS

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
    "WEST":  (-1, 0, BIT_W),
    "SOUTH": (0, -1, BIT_S),
}
OPPOSITE_BIT = {"NORTH": BIT_S, "EAST": BIT_W, "SOUTH": BIT_N, "WEST": BIT_E}

UNIVERSAL_ACTIONS = [
    "NORTH", "EAST", "WEST", "SOUTH",
    "JUMP_NORTH",
    "REMOVE_NORTH", "REMOVE_EAST", "REMOVE_WEST",
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER",
    "TRANSFORM",
    "IDLE",
]
NUM_ACTIONS = len(UNIVERSAL_ACTIONS)

TYPE_MASKS = {
    TYPE_FACTORY: np.array([1,1,1,1, 1, 0,0,0, 1,1,1, 0, 1], dtype=np.float32),
    TYPE_SCOUT:   np.array([1,1,1,1, 0, 0,0,0, 0,0,0, 0, 1], dtype=np.float32),
    TYPE_WORKER:  np.array([1,1,1,1, 0, 1,1,1, 0,0,0, 0, 1], dtype=np.float32),
    TYPE_MINER:   np.array([1,1,1,1, 0, 0,0,0, 0,0,0, 1, 1], dtype=np.float32),
}

GRID_R = 2
WALL_CH = 5
NUM_WALL = (2 * GRID_R + 1) ** 2 * WALL_CH  # 125
NUM_SCALARS = 37
INPUT_SIZE = NUM_WALL + NUM_SCALARS  # 162


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

def friendly_at(occupied, cell, my_player):
    return any(o[1][4] == my_player for o in occupied.get(cell, []))


# ─── NN Forward Pass (numpy) ──────────────────────────────────────────

def nn_forward(features, mask):
    x = features
    x = np.maximum(0, x @ WEIGHTS['backbone.0.weight'].T + WEIGHTS['backbone.0.bias'])
    x = np.maximum(0, x @ WEIGHTS['backbone.2.weight'].T + WEIGHTS['backbone.2.bias'])
    logits = x @ WEIGHTS['policy_head.weight'].T + WEIGHTS['policy_head.bias']
    logits[mask == 0] = -1e9
    logits -= logits.max()
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()
    return np.argmax(probs)


# ─── Feature Extraction ───────────────────────────────────────────────

def _count_types(obs, my_player):
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

def extract_unit(obs, config, my_player, occupied, reserved, uid, data):
    unit_type = data[0]
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    w = config.width
    turn = STATE["turn"]
    gap = r - obs.southBound

    # Wall grid
    grid = np.zeros((5, 5, 5), dtype=np.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 2 + 1):
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
    wall_flat = grid.flatten()

    # Scalars
    sc, wc, mc = _count_types(obs, my_player)
    has_nodes = float(bool(getattr(obs, "miningNodes", None)))
    stuck = STATE.get("factory_stuck", 0) if unit_type == TYPE_FACTORY else 0

    fuid, fdata = _find_factory(obs, my_player)
    if fdata is not None:
        fc, fr = fdata[1], fdata[2]
        dist_to_factory = abs(c - fc) + abs(r - fr)
        dir_to_factory = [
            float(r < fr and c == fc),
            float(c > fc and r == fr),
            float(c < fc and r == fr),
            float(r > fr and c == fc),
        ]
        is_at_spawn = float(c == fc and r == fr + 1)
    else:
        fc, fr = c, r
        dist_to_factory = 0.0
        dir_to_factory = [0.0, 0.0, 0.0, 0.0]
        is_at_spawn = 0.0

    reserved_nearby = sum(1 for dc2 in range(-1, 2) for dr2 in range(-1, 2)
                          if (c + dc2, r + dr2) in reserved)

    # Crystal
    best_cdist = 99.0
    best_cval = 0.0
    for key, val in (getattr(obs, "crystals", None) or {}).items():
        cc2, cr2 = parse_key(key)
        d2 = abs(cc2 - c) + abs(cr2 - r)
        if d2 < best_cdist:
            best_cdist = d2
            best_cval = val

    # Walls near factory
    walls_near_factory = 0
    if fdata is not None and abs(c - fc) + abs(r - fr) <= 2:
        fw = wb(obs, config, c, r)
        if fw is not None:
            walls_near_factory = sum(1 for bit in [BIT_N, BIT_E, BIT_W] if fw & bit)

    # Mining
    visible_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", None) or {}))
    is_on_node = float((c, r) in visible_nodes)
    best_ndist = 99.0
    for key in visible_nodes:
        nc2, nr2 = parse_key(key)
        d2 = abs(nc2 - c) + abs(nr2 - r)
        if d2 < best_ndist:
            best_ndist = d2

    # Enemy
    adj_enemy = 0
    nearest_enemy = 99.0
    enemy_vis = 0
    enemy_factory_vis = 0.0
    for uid2, d2 in obs.robots.items():
        if d2[4] != my_player:
            enemy_vis += 1
            d3 = abs(d2[1] - c) + abs(d2[2] - r)
            if d3 < nearest_enemy:
                nearest_enemy = d3
            if d3 <= 1:
                adj_enemy += 1
            if d2[0] == TYPE_FACTORY:
                enemy_factory_vis = 1.0

    mines = getattr(obs, "mines", None) or {}
    my_mines = sum(1 for v in mines.values() if v[2] == my_player)
    crystals_vis = len(getattr(obs, "crystals", None) or {})

    scalars = np.array([
        float(unit_type == TYPE_FACTORY), float(unit_type == TYPE_SCOUT),
        float(unit_type == TYPE_WORKER), float(unit_type == TYPE_MINER),
        gap / 20.0, energy / 1000.0, move_cd / 5.0,
        c / max(1, w - 1), r / 50.0, turn / 500.0, has_nodes,
        sc / 3.0, wc / 2.0, mc / 2.0,
        stuck / 10.0,
        jump_cd / 20.0, build_cd / 10.0, float(unit_type == TYPE_FACTORY),
        dist_to_factory / 20.0, *dir_to_factory, is_at_spawn, reserved_nearby / 4.0,
        best_cdist / 10.0, best_cval / 50.0,
        walls_near_factory / 4.0, 0.0,
        is_on_node, best_ndist / 10.0,
        adj_enemy / 3.0, nearest_enemy / 10.0, enemy_factory_vis,
        my_mines / 3.0, enemy_vis / 3.0, crystals_vis / 5.0,
    ], dtype=np.float32)

    features = np.concatenate([wall_flat, scalars])

    # Action mask
    mask = TYPE_MASKS[unit_type].copy()

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

    if unit_type == TYPE_FACTORY:
        if jump_cd != 0 or turn <= 2 or not in_bounds(c, r + 2, obs, config):
            mask[4] = 0
        s_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if move_cd == 0 or build_cd != 0 or not s_ok:
            mask[8:11] = 0
        else:
            spawn = (c, r + 1)
            if friendly_at(occupied, spawn, my_player):
                mask[8:11] = 0
            else:
                if energy < getattr(config, "workerCost", 200): mask[8] = 0
                if energy < getattr(config, "scoutCost", 50): mask[9] = 0
                if not has_nodes or energy < getattr(config, "minerCost", 300): mask[10] = 0

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

    if unit_type == TYPE_MINER:
        transform_cost = getattr(config, "transformCost", 100)
        if (c, r) not in visible_nodes or energy < transform_cost + 1:
            mask[11] = 0

    mask[12] = 1.0

    return features, mask


# ─── Main Agent ───────────────────────────────────────────────────────

def agent(obs, config):
    my_player = obs.player
    update_state(obs, config, my_player)

    actions = {}
    reserved = set()
    occupied = {}
    for uid, data in obs.robots.items():
        occupied.setdefault((data[1], data[2]), []).append((uid, data))

    for proc_type in [TYPE_SCOUT, TYPE_WORKER, TYPE_MINER, TYPE_FACTORY]:
        for uid, data in obs.robots.items():
            if uid in actions:
                continue
            if data[4] != my_player or data[0] != proc_type:
                continue

            feat, msk = extract_unit(obs, config, my_player, occupied, reserved, uid, data)
            if feat is None:
                continue

            ai = nn_forward(feat, msk)
            action_str = UNIVERSAL_ACTIONS[ai]

            if action_str in DIRS:
                dc2, dr2, _ = DIRS[action_str]
                reserved.add((data[1] + dc2, data[2] + dr2))
            elif action_str == "JUMP_NORTH":
                reserved.add((data[1], data[2] + 2))
            else:
                reserved.add((data[1], data[2]))

            actions[uid] = action_str

    return actions
