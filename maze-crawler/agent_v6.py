"""v6 agent: strategy-based rewrite with wall memory, mine economy, scroll prediction.

Key principles from game strategy:
1. Mine economy: 50 energy/turn per mine >> crystal collection
2. Wall memory: remember walls permanently for better BFS
3. Scroll speed prediction: danger zone adapts to scroll rate
4. Emergency mode after 400 steps: stop building,全力北移
5. Worker clears path ahead of factory
"""
from collections import deque

# ─── State ────────────────────────────────────────────────────────────────

STATE = {
    "turn": 0,
    "walls": {},          # (col, row) -> wall bitfield, permanent memory
    "nodes": set(),       # discovered mining nodes (col, row)
    "mines": {},          # (col, row) -> [energy, maxEnergy, owner]
    "enemy_pos": {},      # uid -> (col, row, turn)
    "factory_stuck": 0,
    "factory_last_pos": None,
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


# ─── Helpers ──────────────────────────────────────────────────────────────

def parse_key(key):
    c, r = key.split(",")
    return int(c), int(r)


def in_bounds(c, r, obs, config):
    return 0 <= c < config.width and obs.southBound <= r <= obs.northBound


def update_walls(obs, config):
    """Merge visible walls into permanent memory."""
    w = config.width
    for i, v in enumerate(obs.walls):
        if v != -1:
            r = obs.southBound + i // w
            c = i % w
            STATE["walls"][(c, r)] = v


def wb(obs, config, c, r):
    idx = (r - obs.southBound) * config.width + c
    if 0 <= idx < len(obs.walls):
        w = obs.walls[idx]
        if w != -1:
            return w
    return None


def can_go(c, r, d, obs, config):
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
    update_walls(obs, config)
    for key in getattr(obs, "miningNodes", {}) or {}:
        STATE["nodes"].add(parse_key(key))
    for key, data in getattr(obs, "mines", {}).items():
        STATE["mines"][parse_key(key)] = data
    for uid, data in obs.robots.items():
        if data[4] != my_player:
            STATE["enemy_pos"][uid] = (data[1], data[2], STATE["turn"])
        elif data[0] == TYPE_FACTORY:
            pos = (data[1], data[2])
            if STATE["factory_last_pos"] is not None:
                if pos == STATE["factory_last_pos"]:
                    STATE["factory_stuck"] += 1
                else:
                    STATE["factory_stuck"] = 0
            STATE["factory_last_pos"] = pos


# ─── Scroll Speed Prediction ──────────────────────────────────────────────

def scroll_interval(step, config):
    """How many steps between scrolls at current game step."""
    start = getattr(config, 'scrollStartInterval', 4)
    end = getattr(config, 'scrollEndInterval', 1)
    ramp = getattr(config, 'scrollRampSteps', 400)
    if step >= ramp:
        return end
    ratio = step / ramp
    return max(1, int(start - (start - end) * ratio))


def danger_level(factory_row, south_bound, step, config):
    """How many rows of safety margin in next 20 steps."""
    rows_above = factory_row - south_bound
    scrolls = sum(1 for s in range(step, step + 20)
                  if s % max(1, scroll_interval(s, config)) == 0)
    return rows_above - scrolls


# ─── BFS ──────────────────────────────────────────────────────────────────

def bfs_first_step(start, goals, obs, config, max_nodes=500):
    """BFS using wall memory. Returns first direction to reach any goal."""
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
            if not can_go(cur[0], cur[1], d, obs, config):
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


def bfs_to_row(start, row, obs, config):
    goals = [(c, row) for c in range(config.width) if in_bounds(c, row, obs, config)]
    return bfs_first_step(start, goals, obs, config)


def bfs_to_pos(start, goal, obs, config):
    return bfs_first_step(start, [goal], obs, config)


# ─── Movement Helpers ─────────────────────────────────────────────────────

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
    dc, dr, _ = DIRS[d]
    nxt = (c + dc, r + dr)
    if nxt in reserved:
        return False
    occ = occupied.get(nxt, [])
    friendlies = [o for o in occ if o[1][4] == my_player and o[1][0] != TYPE_FACTORY]
    if friendlies:
        all_moving = all(o[0] in actions and actions[o[0]] in DIRS for o in friendlies)
        if not all_moving:
            return False
    actions[uid] = d
    reserved.add(nxt)
    return True


def move_toward(uid, c, r, goals, obs, config, actions, reserved, occupied, my_player):
    """Try BFS toward goals, then greedy fallback."""
    step = bfs_first_step((c, r), goals, obs, config)
    if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
        return True
    return False


def move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row=None):
    if target_row is None:
        target_row = r + 1
    target_row = min(obs.northBound, target_row)
    step = bfs_to_row((c, r), target_row, obs, config)
    if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
        return True
    for d in ["NORTH", "EAST", "WEST", "SOUTH"]:
        if can_go(c, r, d, obs, config):
            if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                return True
    return False


# ─── Factory ──────────────────────────────────────────────────────────────

def factory_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    turn = STATE["turn"]
    stuck = STATE["factory_stuck"]
    width = config.width

    danger = danger_level(r, obs.southBound, turn, config)
    emergency = turn >= 400 or danger <= 2

    scout_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_SCOUT)
    worker_count = sum(1 for d in obs.robots.values()
                       if d[4] == my_player and d[0] == TYPE_WORKER)
    miner_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_MINER)
    has_nodes = bool(getattr(obs, "miningNodes", {})) or bool(STATE["nodes"])

    # ── JUMP ── (danger-aware)
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        should_jump = False
        if gap <= 2:
            should_jump = True
        elif danger <= 3:
            should_jump = True
        elif stuck >= 2:
            should_jump = True
        else:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                if not can_go(c, r, "EAST", obs, config) and not can_go(c, r, "WEST", obs, config):
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
                    if can_go(c, lr, d, obs, config):
                        actions[uid] = "JUMP_NORTH"
                        reserved.add((c, lr))
                        return

    # ── MOVE ──
    if move_cd == 0:
        # Direct NORTH
        if can_go(c, r, "NORTH", obs, config):
            if factory_try_move(uid, c, r, "NORTH", obs, config, actions, reserved, occupied, my_player):
                return

        # BFS to row ahead
        target_row = min(obs.northBound, r + 1)
        step = bfs_to_row((c, r), target_row, obs, config)
        if step:
            dc, dr, _ = DIRS[step]
            if dr >= 0:
                if factory_try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                    return

        # Lateral
        center = width // 4
        ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]
        for d in ew:
            if can_go(c, r, d, obs, config):
                if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return

        # Diagonal
        for d in ew:
            if can_go(c, r, d, obs, config):
                dc, dr, _ = DIRS[d]
                side = (c + dc, r)
                if can_go(side[0], side[1], "NORTH", obs, config):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                        return

        # BFS allowing south when stuck
        if stuck >= 3:
            step = bfs_to_row((c, r), r + 1, obs, config)
            if step:
                if factory_try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                    return

        # SOUTH last resort
        if stuck >= 4 and gap >= 3:
            if can_go(c, r, "SOUTH", obs, config):
                if factory_try_move(uid, c, r, "SOUTH", obs, config, actions, reserved, occupied, my_player):
                    return

    # ── BUILD ──
    if move_cd != 0 and build_cd == 0 and gap >= 2:
        spawn_ok = can_go(c, r, "NORTH", obs, config) and in_bounds(c, r + 1, obs, config)
        if spawn_ok:
            spawn = (c, r + 1)
            if not friendly_at(occupied, spawn, my_player):
                worker_cost = getattr(config, "workerCost", 200)

                # Worker for wall clearing
                if worker_count < 1 and energy >= worker_cost + 100 and gap >= 2:
                    actions[uid] = "BUILD_WORKER"
                    reserved.add(spawn)
                    return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Scout ────────────────────────────────────────────────────────────────

def scout_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Energy transfer to nearby friendly when nearly full
    max_e = getattr(config, 'scoutMaxEnergy', 100)
    if energy >= max_e * 0.8:
        for d in ("NORTH", "EAST", "WEST", "SOUTH"):
            dc, dr, _ = DIRS[d]
            nc, nr = c + dc, r + dr
            if can_go(c, r, d, obs, config):
                for occ_uid, occ_data in occupied.get((nc, nr), []):
                    if occ_data[4] == my_player:
                        actions[uid] = f"TRANSFER_{d}"
                        reserved.add((c, r))
                        return

    # Find nearest mining node (for Miner economy)
    vis_nodes = [parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {})]
    if vis_nodes:
        nearest = min(vis_nodes, key=lambda n: abs(n[0] - c) + abs(n[1] - r))
        step = bfs_to_pos((c, r), nearest, obs, config)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return

    # Collect crystals
    crystals = [(parse_key(k), v) for k, v in (getattr(obs, "crystals", {}) or {}).items()]
    if crystals:
        best = max(crystals, key=lambda cv: cv[1] / max(1, abs(cv[0][0] - c) + abs(cv[0][1] - r)))
        step = bfs_to_pos((c, r), best[0], obs, config)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return

    # Explore ahead of factory
    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break
    if factory_pos:
        target_row = min(obs.northBound, factory_pos[1] + 6)
        if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row):
            return

    # Default north
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Worker ───────────────────────────────────────────────────────────────

def worker_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    wall_cost = getattr(config, "wallRemoveCost", 100)

    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break

    # Escape factory's north cell
    if factory_pos and (c, r) == (factory_pos[0], factory_pos[1] + 1):
        for d in ("NORTH", "EAST", "WEST"):
            if can_go(c, r, d, obs, config):
                if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return

    # Remove walls near factory path
    if factory_pos and energy >= wall_cost:
        fc, fr = factory_pos
        # On factory's north path: clear north walls
        if c == fc and r >= fr:
            w = wb(obs, config, c, r) or 0
            if w & BIT_N:
                actions[uid] = "REMOVE_NORTH"
                reserved.add((c, r))
                return
        # Near factory: clear any blocking wall
        if abs(c - fc) + abs(r - fr) <= 2:
            w = wb(obs, config, c, r) or 0
            for d, bit in [("NORTH", BIT_N), ("EAST", BIT_E), ("WEST", BIT_W)]:
                if w & bit:
                    actions[uid] = f"REMOVE_{d}"
                    reserved.add((c, r))
                    return

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Follow factory path
    target_row = r + 1
    if factory_pos:
        target_row = min(obs.northBound, factory_pos[1] + 2)
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player, target_row):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Miner ────────────────────────────────────────────────────────────────

def miner_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    transform_cost = getattr(config, "transformCost", 100)

    # TRANSFORM on mining node immediately
    vis_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {}))
    if (c, r) in vis_nodes and energy >= transform_cost + 1:
        actions[uid] = "TRANSFORM"
        reserved.add((c, r))
        return

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Find nearest mining node
    mines = set(parse_key(k) for k in getattr(obs, "mines", {}).keys())
    vis_list = [n for n in vis_nodes if n not in mines]
    rem_list = [n for n in STATE["nodes"] if n not in mines and in_bounds(n[0], n[1], obs, config)]
    all_nodes = vis_list + rem_list

    if all_nodes:
        target = min(all_nodes, key=lambda n: abs(n[0] - c) + abs(n[1] - r))
        step = bfs_first_step((c, r), [target], obs, config)
        if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
            return

    # No nodes: follow factory
    if move_north(uid, c, r, obs, config, actions, reserved, occupied, my_player):
        return

    actions[uid] = "IDLE"
    reserved.add((c, r))


# ─── Main ─────────────────────────────────────────────────────────────────

def agent(obs, config):
    my_player = obs.player
    update_state(obs, config, my_player)

    actions = {}
    reserved = set()
    occupied = {}
    for uid, data in obs.robots.items():
        occupied.setdefault((data[1], data[2]), []).append((uid, data))

    # Process non-factory units first so they can escape factory's path
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_SCOUT:
            scout_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_WORKER:
            worker_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_MINER:
            miner_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    # Factory last
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            factory_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    return actions
