"""v2 agent: BFS pathfinding + aggressive JUMP, no workers/miners."""
from collections import deque

STATE = {"turn": 0, "walls": {}}

TYPE_FACTORY = 0
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
    """Optimistic: unknown = passable."""
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
    """Pessimistic: unknown = wall."""
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


def bfs_first_step(start, goals, obs, config, passable_fn, max_nodes=500):
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


def agent(obs, config):
    STATE["turn"] += 1
    actions = {}
    width = config.width

    for uid, data in obs.robots.items():
        if data[4] != obs.player or data[0] != TYPE_FACTORY:
            continue
        c, r, energy = data[1], data[2], data[3]
        move_cd = data[5] if len(data) > 5 else 0
        jump_cd = data[6] if len(data) > 6 else 0
        gap = r - obs.southBound

        # JUMP: always when available
        if jump_cd == 0 and in_bounds(c, r + 2, obs, config):
            actions[uid] = "JUMP_NORTH"
            return actions

        # MOVE
        if move_cd == 0:
            # BFS to row+2 goals
            goals = [(c2, r + 2) for c2 in range(width) if in_bounds(c2, r + 2, obs, config)]
            
            # Tier 1: Direct NORTH
            if can_go(obs, config, c, r, "NORTH"):
                actions[uid] = "NORTH"
                return actions

            # Tier 2: BFS (optimistic)
            step = bfs_first_step((c, r), goals, obs, config, can_go, max_nodes=500)
            if step:
                dc, dr, _ = DIRS[step]
                if dr >= 0:  # NORTH, EAST, or WEST only (no south via BFS)
                    actions[uid] = step
                    return actions

            # Tier 3: Lateral
            center = width // 2
            if c < center:
                ew = ["EAST", "WEST"]
            else:
                ew = ["WEST", "EAST"]
            for d in ew:
                if can_go(obs, config, c, r, d):
                    actions[uid] = d
                    return actions

            # Tier 4: BFS pessimistic
            step = bfs_first_step((c, r), goals, obs, config, can_go_pessimistic, max_nodes=500)
            if step:
                actions[uid] = step
                return actions

            # Tier 5: SOUTH as absolute last resort
            if can_go(obs, config, c, r, "SOUTH"):
                actions[uid] = "SOUTH"
                return actions

        actions[uid] = "IDLE"

    return actions
