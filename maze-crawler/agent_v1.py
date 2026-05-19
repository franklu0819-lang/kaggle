"""v9 agent: fast exploration + dynamic strategy.

Key principles:
1. Early game: build scouts immediately for fast maze exploration
2. Scouts reveal walls ahead of factory, enabling better pathfinding
3. Mid game: build worker to clear walls, miner for mine economy
4. Late game: pure survival
5. Strategy adapts to gap/energy/turn dynamically
"""
from collections import deque

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
    """Optimistic: unknown = passable, only known walls block."""
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


def factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
    """Factory movement: ignores friendly units (crushes them), only checks walls and reserved targets."""
    dc, dr, _ = DIRS[d]
    nxt = (c + dc, r + dr)
    if nxt in reserved:
        return False
    # Don't crush our own units if they have no escape (wasteful)
    # But DO move if the cell is clear or has only enemies (crush them)
    occ = occupied.get(nxt, [])
    friendlies = [o for o in occ if o[1][4] == my_player and o[1][0] != TYPE_FACTORY]
    if friendlies:
        # Check if ALL friendlies there have been assigned a move action (will move away)
        all_moving = all(o[0] in actions and actions[o[0]] in DIRS for o in friendlies)
        if not all_moving:
            return False  # Would crush our own unit that isn't moving
    actions[uid] = d
    reserved.add(nxt)
    return True


def move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row=None):
    """Non-factory unit: move toward north using BFS + greedy fallback."""
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


# ─── Factory ─────────────────────────────────────────────────────────────

def factory_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    turn = STATE["turn"]
    stuck = STATE["factory_stuck"]
    width = config.width

    # Count units
    scout_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_SCOUT)
    worker_count = sum(1 for d in obs.robots.values()
                       if d[4] == my_player and d[0] == TYPE_WORKER)
    miner_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_MINER)
    my_mines = sum(1 for k, v in getattr(obs, "mines", {}).items() if v[2] == my_player)

    # ── JUMP ──
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        should_jump = False
        if gap <= 2:
            should_jump = True
        elif stuck >= 2:
            should_jump = True
        else:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                if not can_go(obs, config, c, r, "EAST") and not can_go(obs, config, c, r, "WEST"):
                    should_jump = True

        if should_jump:
            lr = r + 2
            landing = wb(obs, config, c, lr)
            if landing is None:
                actions[uid] = "JUMP_NORTH"
                reserved.add((c, lr))
                return
            else:
                for d in ("NORTH", "EAST", "WEST", "SOUTH"):
                    if can_go(obs, config, c, lr, d):
                        actions[uid] = "JUMP_NORTH"
                        reserved.add((c, lr))
                        return

    # ── MOVE ── (uses factory_try_move which handles friendly crushing)
    if move_cd == 0:
        center = width // 4
        ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]

        # Tier 1: Direct NORTH if no known wall
        if can_go(obs, config, c, r, "NORTH"):
            if factory_try_move(uid, c, r, "NORTH", obs, config, actions, reserved, occupied, my_player):
                return

        # Tier 2: BFS to next row, but only accept non-south first steps
        step = bfs_to_row((c, r), r + 1, obs, config, can_go)
        if step:
            dc, dr, _ = DIRS[step]
            if dr >= 0:  # NORTH, EAST, or WEST only
                if factory_try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                    return

        # Tier 3: Forced lateral — try EAST/WEST even if BFS didn't find them
        for d in ew:
            if can_go(obs, config, c, r, d):
                if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return

        # Tier 4: Diagonal — go EAST/WEST if the cell north of that is open
        for d in ew:
            if can_go(obs, config, c, r, d):
                dc, dr, _ = DIRS[d]
                side = (c + dc, r)
                if can_go(obs, config, side[0], side[1], "NORTH"):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                        return

        # Tier 5: BFS allowing south (when stuck >= 3)
        if stuck >= 3:
            step = bfs_to_row((c, r), r + 1, obs, config, can_go)
            if step:
                if factory_try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                    return

        # Tier 6: SOUTH as last resort
        if stuck >= 4 and gap >= 3:
            if can_go(obs, config, c, r, "SOUTH"):
                if factory_try_move(uid, c, r, "SOUTH", obs, config, actions, reserved, occupied, my_player):
                    return

    # ── BUILD (during move cooldown) ──
    if move_cd != 0 and build_cd == 0 and gap >= 2:
        spawn_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if spawn_ok:
            spawn = (c, r + 1)
            if not friendly_at(occupied, spawn, my_player):
                scout_cost = getattr(config, "scoutCost", 50)
                worker_cost = getattr(config, "workerCost", 200)
                miner_cost = getattr(config, "minerCost", 300)

                # Priority 1: Build worker for wall clearing
                if worker_count < 1 and energy >= worker_cost + 100 and gap >= 3:
                    actions[uid] = "BUILD_WORKER"
                    reserved.add(spawn)
                    return

                # Priority 2: Build scout for exploration
                if scout_count < 2 and energy >= scout_cost + 100 and gap >= 4 and turn <= 40:
                    actions[uid] = "BUILD_SCOUT"
                    reserved.add(spawn)
                    return

                # Priority 3: Build miner if mining nodes found
                has_nodes = bool(getattr(obs, "miningNodes", {})) or bool(STATE["nodes"])
                if (has_nodes and miner_count < 1 and energy >= miner_cost + 200
                        and gap >= 6 and turn < 150):
                    actions[uid] = "BUILD_MINER"
                    reserved.add(spawn)
                    return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Worker ─────────────────────────────────────────────────────────────

def worker_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    gap = r - obs.southBound
    wall_cost = getattr(config, "wallRemoveCost", 100)

    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break

    # Escape factory's north cell
    if factory_pos and (c, r) == (factory_pos[0], factory_pos[1] + 1):
        for d in ("NORTH", "EAST", "WEST"):
            if can_go(obs, config, c, r, d):
                if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return

    # Remove walls blocking factory's north path
    if factory_pos and energy >= wall_cost + 20:
        fc, fr = factory_pos
        # At factory's north cell: remove north wall (opens path further north)
        if c == fc and r == fr + 1:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                actions[uid] = "REMOVE_NORTH"
                reserved.add((c, r))
                return
        # Near factory: remove wall that blocks factory's lateral movement
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

    # Follow factory
    target_row = r + 1
    if factory_pos:
        target_row = min(obs.northBound, factory_pos[1] + 2)
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Miner ──────────────────────────────────────────────────────────────

def miner_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    transform_cost = getattr(config, "transformCost", 100)

    # TRANSFORM on visible mining node
    visible_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {}))
    if (c, r) in visible_nodes and energy >= transform_cost + 1:
        actions[uid] = "TRANSFORM"
        reserved.add((c, r))
        return

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Find mining node
    mines = set(parse_key(k) for k in getattr(obs, "mines", {}).keys())
    vis_list = [n for n in visible_nodes if n not in mines]
    rem_list = [n for n in STATE["nodes"] if n not in mines and in_bounds(n[0], n[1], obs, config)]
    all_nodes = vis_list + rem_list

    if all_nodes:
        target = min(all_nodes, key=lambda n: abs(n[0] - c) + abs(n[1] - r))
        step = bfs_first_step((c, r), [target], obs, config, can_go)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return

    # Follow factory
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Scout ──────────────────────────────────────────────────────────────

def scout_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    gap = r - obs.southBound

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Collect nearby crystals
    crystals = [(parse_key(k), v) for k, v in (getattr(obs, "crystals", {}) or {}).items()]
    if crystals:
        best = max(
            [(v / max(1, abs(cell[0] - c) + abs(cell[1] - r)), cell)
             for cell, v in crystals if cell != (c, r)],
            key=lambda x: x[0],
            default=None,
        )
        if best:
            _, target = best
            step = bfs_first_step((c, r), [target], obs, config, can_go)
            if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                return

    # Explore ahead of factory
    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break

    if factory_pos:
        fc, fr = factory_pos
        # Scout goes 5-8 cells ahead
        target_row = min(obs.northBound, fr + 6)
        # Spread scouts: alternate between center-ish and side exploration
        scout_idx = sum(1 for d in obs.robots.values()
                        if d[4] == my_player and d[0] == TYPE_SCOUT and d[1] == c and d[2] == r)
        half = config.width // 2
        if c < half:
            target_col = min(half - 1, c + 3)
        else:
            target_col = max(half, c - 3)
        step = bfs_first_step((c, r), [(target_col, target_row)], obs, config, can_go)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return

    # Default: move north
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Main ────────────────────────────────────────────────────────────────

def agent(obs, config):
    my_player = obs.player
    update_state(obs, config, my_player)

    actions = {}
    reserved = set()
    occupied = {}
    for uid, data in obs.robots.items():
        cell = (data[1], data[2])
        occupied.setdefault(cell, []).append((uid, data))

    # Process non-factory units FIRST so they escape the factory's path
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_SCOUT:
            scout_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_WORKER:
            worker_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_MINER:
            miner_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    # Factory LAST: now units have their actions, factory can safely move through
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            factory_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    return actions
