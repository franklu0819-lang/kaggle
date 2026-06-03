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
    "walls": {},
    "mine_invested": None,
    "mine_wait": False,       # True after BUILD_MINER, wait for mine to appear
    "mine_wait_since": 0,    # turn when we started waiting
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
            STATE["walls"][(c, r)] = w
            return w
    return STATE["walls"].get((c, r))


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


def can_go_pessimistic(obs, config, c, r, d):
    """Pessimistic: unknown cells treated as walls."""
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


def update_state(obs, config, my_player):
    STATE["turn"] += 1
    for key in getattr(obs, "miningNodes", {}) or {}:
        STATE["nodes"].add(parse_key(key))
    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            pos = (data[1], data[2])
            if STATE["last_factory_pos"] is not None:
                if pos[1] <= STATE["last_factory_pos"][1]:
                    STATE["factory_stuck"] += 1
                else:
                    STATE["factory_stuck"] = 0
            STATE["last_factory_pos"] = pos
            break


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


def bfs_to_row(start, row, obs, config, passable_fn):
    goals = [(c, row) for c in range(config.width) if in_bounds(c, row, obs, config)]
    return bfs_first_step(start, goals, obs, config, passable_fn)


def bfs_distance(start, goal, obs, config, passable_fn, max_nodes=500):
    """Return BFS distance from start to goal, or None if unreachable."""
    if start == goal:
        return 0
    q = deque([(start, 0)])
    visited = {start}
    while q:
        cur, dist = q.popleft()
        for d in ("NORTH", "EAST", "WEST", "SOUTH"):
            if not passable_fn(obs, config, cur[0], cur[1], d):
                continue
            dc, dr, _ = DIRS[d]
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt in visited:
                continue
            if nxt == goal:
                return dist + 1
            visited.add(nxt)
            q.append((nxt, dist + 1))
            if len(visited) >= max_nodes:
                return None
    return None


def calc_mine_roi(mine_node, factory_c, factory_r, gap, step, obs, config):
    """Calculate expected energy output from investing in a mine node.
    Returns expected_output (energy), or 0 if not viable."""
    mc, mr = mine_node
    if mr < factory_r or not in_bounds(mc, mr, obs, config):
        return 0
    dist = bfs_distance((factory_c, factory_r), mine_node, obs, config, can_go)
    if dist is None:
        return 0
    turns_to_reach = dist * 2  # factory moves every 2 turns
    start_int = getattr(config, "scrollStartInterval", 4)
    end_int = getattr(config, "scrollEndInterval", 1)
    ramp_steps = getattr(config, "scrollRampSteps", 400)
    progress = min(1.0, step / ramp_steps)
    scroll_interval = max(float(end_int), start_int - (start_int - end_int) * progress)
    gap_at_arrival = gap + dist - turns_to_reach / scroll_interval
    stay_turns = gap_at_arrival - 2  # safety margin
    effective_stay = max(0, stay_turns - 3)  # build + move + TRANSFORM overhead
    return effective_stay * 50


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


def get_enemy_factory_threat(obs, config, my_player):
    """Return (hard_block, danger) for enemy factory threat avoidance.

    `hard_block` — cells we must NEVER enter under any circumstance:
        - every enemy factory's current cell
      Entering one guarantees a mutual-destruction collision. The post-collision
      tiebreaker (total team energy) empirically goes against us, so we treat
      these cells as walls regardless of scroll pressure.

    `danger` — cells the enemy factory could occupy NEXT turn:
        - hard_block (they may IDLE)
        - if their move_cd==0: 4 MOVE neighbors that pass `can_go`
        - if their jump_cd==0: 4 JUMP_N/S/E/W landings (jumps ignore walls)
      Cooldown gating is critical: without it the danger zone is so pessimistic
      that whenever an enemy factory is in vision the agent retreats and
      oscillates instead of pushing north (regressed otherwise-winnable seeds
      like 1138 from a 223-step win to a 442-step scroll-out loss). With the
      gate, danger only fires when the enemy can actually act this turn.
    """
    hard_block = set()
    danger = set()
    for uid, d in obs.robots.items():
        if d[4] == my_player or d[0] != TYPE_FACTORY:
            continue
        ec, er = d[1], d[2]
        emcd = d[5] if len(d) > 5 else 0
        ejcd = d[6] if len(d) > 6 else 0
        # Hard block: enemy factory current cell — never enter (mutual destruct).
        hard_block.add((ec, er))
        danger.add((ec, er))
        # MOVE neighbors only if enemy can move THIS turn (move_cd == 0)
        if emcd == 0:
            for d_str in ("NORTH", "EAST", "WEST", "SOUTH"):
                if can_go(obs, config, ec, er, d_str):
                    dc, dr, _ = DIRS[d_str]
                    danger.add((ec + dc, er + dr))
        # JUMP landings only if enemy can jump THIS turn (jump_cd == 0)
        if ejcd == 0:
            for jdc, jdr in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                lc, lr = ec + jdc, er + jdr
                if in_bounds(lc, lr, obs, config):
                    danger.add((lc, lr))
    return hard_block, danger


def factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                     allow_crush=False, danger=None, allow_danger=False, hard_block=None):
    """Factory movement: ignores friendly units (crushes them), only checks walls and reserved targets.

    `hard_block` cells (e.g. enemy factory current cells) are ALWAYS rejected.
    `danger` cells are rejected unless `allow_danger=True` (panic mode).
    """
    dc, dr, _ = DIRS[d]
    nxt = (c + dc, r + dr)
    if nxt in reserved:
        return False
    if hard_block is not None and nxt in hard_block:
        return False
    if danger is not None and nxt in danger and not allow_danger:
        return False
    occ = occupied.get(nxt, [])
    friendlies = [o for o in occ if o[1][4] == my_player and o[1][0] != TYPE_FACTORY]
    if friendlies and not allow_crush:
        all_moving = all(o[0] in actions and actions[o[0]] in DIRS for o in friendlies)
        if not all_moving:
            return False
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

    # ── Enemy factory threat zones ──
    # `enemy_hard_block` — enemy factory current cells. Always avoid: entering
    #   one is a guaranteed mutual-destruction → losing tiebreaker.
    # `enemy_danger`     — cells enemy factory could reach next turn (move +
    #   jump). Avoid when possible; accept under scroll pressure (gap≤3) or
    #   when our own cell is already in the zone.
    enemy_hard_block, enemy_danger = get_enemy_factory_threat(obs, config, my_player)

    # Count units
    scout_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_SCOUT)
    worker_count = sum(1 for d in obs.robots.values()
                       if d[4] == my_player and d[0] == TYPE_WORKER)
    miner_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_MINER)
    my_mines = sum(1 for k, v in getattr(obs, "mines", {}).items() if v[2] == my_player)

    # ── Mine target selection (dynamic ROI based on gap, scroll speed, energy) ──
    _start_int = getattr(config, "scrollStartInterval", 4)
    _end_int = getattr(config, "scrollEndInterval", 1)
    _ramp_steps = getattr(config, "scrollRampSteps", 400)
    _progress = min(1.0, turn / _ramp_steps)
    _scroll_interval = max(float(_end_int), _start_int - (_start_int - _end_int) * _progress)
    panic_steps = gap * _scroll_interval
    if panic_steps >= 100:
        roi_threshold = 50
    elif panic_steps >= 50:
        roi_threshold = 100
    elif panic_steps >= 25:
        roi_threshold = 200
    else:
        roi_threshold = 9999

    mine_target = None
    if STATE["mine_invested"]:
        mn = STATE["mine_invested"]
        if in_bounds(mn[0], mn[1], obs, config):
            roi = calc_mine_roi(mn, c, r, gap, turn, obs, config)
            if roi >= roi_threshold:
                mine_target = mn
        if mine_target is None:
            STATE["mine_invested"] = None

    if mine_target is None:
        existing_mines = set(parse_key(k) for k in getattr(obs, "mines", {}).keys())
        candidates = []
        for node in STATE["nodes"]:
            if node in existing_mines:
                continue
            if node[1] < r or not in_bounds(node[0], node[1], obs, config):
                continue
            roi = calc_mine_roi(node, c, r, gap, turn, obs, config)
            if roi >= roi_threshold:
                d = abs(node[0] - c) + abs(node[1] - r)
                candidates.append((d, node))
        if candidates:
            candidates.sort()
            mine_target = candidates[0][1]
            STATE["mine_invested"] = mine_target

    # ── JUMP ── (aggressive: always JUMP when available, like v2)
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        # Try JUMP_NORTH first. Only avoid landing on enemy factory cell.
        lr = r + 2
        if (in_bounds(c, lr, obs, config)
                and (c, lr) not in enemy_hard_block):
            actions[uid] = "JUMP_NORTH"
            reserved.add((c, lr))
            return

        # Lateral jumps as fallback
        for jd, (jdc, jdr) in (("JUMP_EAST", (2, 0)), ("JUMP_WEST", (-2, 0))):
            lc, lr2 = c + jdc, r + jdr
            if not in_bounds(lc, lr2, obs, config):
                continue
            if (lc, lr2) in enemy_hard_block:
                continue
            actions[uid] = jd
            reserved.add((lc, lr2))
            return

    # ── MOVE ── (simple tiers, danger avoidance to prevent factory collisions)
    if move_cd == 0:
        # ── Mine wait/collection removed: factory never IDLEs for mines ──
        if STATE["mine_wait"]:
            STATE["mine_wait"] = False
            STATE["mine_invested"] = None

        center = width // 4
        ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]

        # BFS goals: navigate northward to row+2
        north_goals = [(c2, r + 2) for c2 in range(width) if in_bounds(c2, r + 2, obs, config)]

        # Danger avoidance: avoid enemy factory danger zone unless scroll pressure is high
        panic = (gap <= 3) or (c, r) in enemy_danger

        def _move_ok(d):
            dc, dr, _ = DIRS[d]
            nxt = (c + dc, r + dr)
            if nxt in enemy_hard_block:
                return False
            if nxt in enemy_danger and not panic:
                return False
            return True

        # Tier 1: Direct NORTH if no known wall
        if can_go(obs, config, c, r, "NORTH") and _move_ok("NORTH"):
            dc, dr, _ = DIRS["NORTH"]
            actions[uid] = "NORTH"
            reserved.add((c + dc, r + dr))
            return

        # Tier 2: BFS to row+2 (optimistic)
        bfs_limit = 500
        step_dir = bfs_first_step((c, r), north_goals, obs, config, can_go, max_nodes=bfs_limit)
        if step_dir:
            dc2, dr2, _ = DIRS[step_dir]
            if dr2 >= 0 and _move_ok(step_dir):
                actions[uid] = step_dir
                reserved.add((c + dc2, r + dr2))
                return

        # Tier 3: Lateral
        for d in ew:
            if can_go(obs, config, c, r, d) and _move_ok(d):
                dc, dr, _ = DIRS[d]
                actions[uid] = d
                reserved.add((c + dc, r + dr))
                return

        # Tier 4: BFS (pessimistic) when stuck
        if stuck >= 2:
            step_dir = bfs_first_step((c, r), north_goals, obs, config, can_go_pessimistic, max_nodes=bfs_limit)
            if step_dir:
                dc2, dr2, _ = DIRS[step_dir]
                if dr2 >= 0 and _move_ok(step_dir):
                    actions[uid] = step_dir
                    reserved.add((c + dc2, r + dr2))
                    return

        # Tier 5: Panic fallback — accept danger cells to escape scroll
        if panic:
            for d in ("NORTH", "EAST", "WEST", "SOUTH"):
                if can_go(obs, config, c, r, d):
                    dc, dr, _ = DIRS[d]
                    nxt = (c + dc, r + dr)
                    if nxt not in enemy_hard_block:
                        actions[uid] = d
                        reserved.add(nxt)
                        return

        # Tier 6: SOUTH as absolute last resort
        if stuck >= 4 and gap >= 3:
            if can_go(obs, config, c, r, "SOUTH"):
                dc, dr, _ = DIRS["SOUTH"]
                nxt = (c + dc, r + dr)
                if nxt not in enemy_hard_block:
                    actions[uid] = "SOUTH"
                    reserved.add(nxt)
                    return

    # ── BUILD (during move cooldown) ──
    if move_cd != 0 and build_cd == 0 and gap >= 2:
        spawn_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if spawn_ok:
            spawn = (c, r + 1)
            if not friendly_at(occupied, spawn, my_player):
                # Build Miner: factory must be at (mc, mr-1) so miner spawns ON the node
                if mine_target and energy >= 600:
                    mc, mr = mine_target
                    if (c, r) == (mc, mr - 1) and spawn_ok:
                        has_miner = any(
                            d2[4] == my_player and d2[0] == TYPE_MINER
                            for d2 in obs.robots.values()
                        )
                        if not has_miner:
                            actions[uid] = "BUILD_MINER"
                            STATE["mine_wait"] = True
                            STATE["mine_wait_since"] = turn
                            reserved.add(spawn)
                            return

                # Build Worker: only when energy is abundant and factory is safe
                if energy >= 1500 and gap >= 8:
                    max_workers = 1
                    if worker_count < max_workers:
                        actions[uid] = "BUILD_WORKER"
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
    factory_uid = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            factory_uid = uid2
            break

    # Transfer energy back to factory when about to die
    if gap <= 1 and factory_pos and energy > 5:
        fc, fr = factory_pos
        for d, (dc, dr, _) in [("NORTH", DIRS["NORTH"]), ("SOUTH", DIRS["SOUTH"]),
                                 ("EAST", DIRS["EAST"]), ("WEST", DIRS["WEST"])]:
            if (c + dc, r + dr) == (fc, fr) and can_go(obs, config, c, r, d):
                actions[uid] = f"TRANSFER_{d}"
                reserved.add((c, r))
                return

    # Escape factory's north cell
    if factory_pos and (c, r) == (factory_pos[0], factory_pos[1] + 1):
        for d in ("NORTH", "EAST", "WEST"):
            if can_go(obs, config, c, r, d):
                if try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player):
                    return

    # Remove walls blocking factory's north path
    if factory_pos and energy >= wall_cost + 20:
        fc, fr = factory_pos
        if c == fc and r == fr + 1:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                actions[uid] = "REMOVE_NORTH"
                reserved.add((c, r))
                return
        # Clear NORTH walls up to 4 rows ahead, lateral walls within 2
        if abs(c - fc) <= 2 and 0 < (r - fr) <= 4:
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

    # Help stuck factory: navigate to factory's north cell to clear blocking wall
    if factory_pos and STATE.get("factory_stuck", 0) >= 2 and energy >= wall_cost + 20:
        fc, fr = factory_pos
        north_cell = (fc, fr + 1)
        if (c, r) != north_cell and in_bounds(north_cell[0], north_cell[1], obs, config):
            step = bfs_first_step((c, r), [north_cell], obs, config, can_go)
            if step and try_move(uid, c, r, step, obs, config, actions, reserved, occupied, my_player):
                return

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

    # Low energy + no walls to clear: stop following factory (die naturally, save maintenance)
    if energy < 30 and factory_pos:
        fc, fr = factory_pos
        nearby_walls = False
        for d, bit in [("NORTH", BIT_N), ("EAST", BIT_E), ("WEST", BIT_W), ("SOUTH", BIT_S)]:
            w = wb(obs, config, c, r)
            if w is not None and (w & bit):
                nearby_walls = True
                break
        if not nearby_walls:
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
        target_row = min(obs.northBound, fr + 6)
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
