"""Minimal agent: just MOVE_NORTH and JUMP_NORTH, nothing else."""
from collections import deque

STATE = {"turn": 0, "walls": {}}

TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER = 0, 1, 2, 3
BIT_N, BIT_E, BIT_S, BIT_W = 1, 2, 4, 8

DIRS = {
    "NORTH": (0, 1, BIT_N),
    "EAST":  (1, 0, BIT_E),
    "SOUTH": (0, -1, BIT_S),
    "WEST":  (-1, 0, BIT_W),
}
OPPOSITE_BIT = {"NORTH": BIT_S, "EAST": BIT_W, "SOUTH": BIT_N, "WEST": BIT_E}


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
            STATE["walls"][(c, r)] = w
            return w
    return STATE["walls"].get((c, r))


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


def can_go_pessimistic(obs, config, c, r, d):
    dc, dr, bit = DIRS[d]
    nc, nr = c + dc, r + dr
    if not in_bounds(nc, nr, obs, config):
        return False
    w = wb(obs, config, c, r)
    if w is None or (w & bit):
        return False
    w2 = wb(obs, config, nc, nr)
    if w2 is None or (w2 & OPPOSITE_BIT[d]):
        return False
    return True


def agent(obs, config):
    STATE["turn"] += 1
    actions = {}
    for uid, data in obs.robots.items():
        if data[4] != obs.player or data[0] != TYPE_FACTORY:
            continue
        c, r = data[1], data[2]
        move_cd = data[5] if len(data) > 5 else 0
        jump_cd = data[6] if len(data) > 6 else 0
        gap = r - obs.southBound

        # JUMP: always when available
        if jump_cd == 0 and in_bounds(c, r + 2, obs, config):
            actions[uid] = "JUMP_NORTH"
            return actions

        # MOVE: north first, then lateral
        if move_cd == 0:
            if can_go(obs, config, c, r, "NORTH"):
                actions[uid] = "NORTH"
                return actions
            # Try EAST or WEST
            for d in ["EAST", "WEST"]:
                if can_go(obs, config, c, r, d):
                    actions[uid] = d
                    return actions
            # Try pessimistic
            for d in ["NORTH", "EAST", "WEST"]:
                if can_go_pessimistic(obs, config, c, r, d):
                    actions[uid] = d
                    return actions
            # SOUTH as absolute last resort
            if can_go(obs, config, c, r, "SOUTH"):
                actions[uid] = "SOUTH"
                return actions

        actions[uid] = "IDLE"
    return actions
