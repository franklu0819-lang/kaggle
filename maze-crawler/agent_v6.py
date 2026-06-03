"""v6 agent: earlier lateral movement when stuck, on v1 base.

Key changes:
- When stuck>=1: expand BFS goals from row+2 to rows r+1..r+4 (find longer lateral detours)
- When stuck>=2: add lateral-escape BFS (find nearest cell with open north path)
- Keep all south movement tiers exactly as v1
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


def bfs_distance(start, goal, obs, config, passable_fn, max_nodes=300):
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
    if panic_steps >= 100 and energy >= 400:
        roi_threshold = 100
    elif panic_steps >= 50 and energy >= 300:
        roi_threshold = 200
    elif panic_steps >= 25:
        roi_threshold = 400
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

    # ── JUMP ──
    if jump_cd == 0 and turn > 2 and in_bounds(c, r + 2, obs, config):
        # Pre-compute open MOVE targets (for danger-escape detection)
        move_targets = []
        for d_str in ("NORTH", "EAST", "WEST", "SOUTH"):
            if can_go(obs, config, c, r, d_str):
                dc_t, dr_t, _ = DIRS[d_str]
                move_targets.append((c + dc_t, r + dr_t))
        # All MOVE options dangerous → JUMP is the only safe escape
        danger_escape = bool(move_targets) and all(t in enemy_danger for t in move_targets)

        should_jump = False
        if gap <= 2:
            should_jump = True
        elif stuck >= 2:
            should_jump = True
        elif danger_escape:
            should_jump = True
        else:
            w = wb(obs, config, c, r)
            if w is not None and (w & BIT_N):
                if not can_go(obs, config, c, r, "EAST") and not can_go(obs, config, c, r, "WEST"):
                    should_jump = True

        if should_jump:
            # Only land in enemy danger as a true last resort
            allow_danger_jump = (gap <= 3)

            # Try JUMP_NORTH first. NEVER land on enemy factory cell (hard_block).
            lr = r + 2
            if (in_bounds(c, lr, obs, config)
                    and (c, lr) not in enemy_hard_block
                    and ((c, lr) not in enemy_danger or allow_danger_jump)):
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

            # Lateral jumps: emergency (gap≤3) OR danger escape
            if gap <= 3 or danger_escape:
                for jd, (jdc, jdr) in (("JUMP_EAST", (2, 0)), ("JUMP_WEST", (-2, 0))):
                    lc, lr2 = c + jdc, r + jdr
                    if not in_bounds(lc, lr2, obs, config):
                        continue
                    if (lc, lr2) in enemy_hard_block:
                        continue  # never crash into enemy factory
                    if (lc, lr2) in enemy_danger and not allow_danger_jump:
                        continue
                    landing = wb(obs, config, lc, lr2)
                    if landing is None:
                        actions[uid] = jd
                        reserved.add((lc, lr2))
                        return
                    else:
                        for d in ("NORTH", "EAST", "WEST"):
                            if can_go(obs, config, lc, lr2, d):
                                actions[uid] = jd
                                reserved.add((lc, lr2))
                                return

    # ── MOVE ── (uses factory_try_move which handles friendly crushing)
    if move_cd == 0:
        # ── Check if we should stay at an existing mine ──
        my_mines_nearby = []
        for mk, mv in getattr(obs, "mines", {}).items():
            mc2, mr2 = parse_key(mk)
            if mv[2] == my_player and abs(mc2 - c) + abs(mr2 - r) <= 1:
                my_mines_nearby.append((mc2, mr2))

        # ── Check mine_wait: waiting for miner to TRANSFORM ──
        if STATE["mine_wait"]:
            mine_exists_nearby = any(
                mv[2] == my_player and abs(parse_key(mk)[0] - c) + abs(parse_key(mk)[1] - r) <= 1
                for mk, mv in getattr(obs, "mines", {}).items()
            )
            waited = turn - STATE["mine_wait_since"]
            if mine_exists_nearby:
                STATE["mine_wait"] = False
                # Don't return — fall through to my_mines_nearby collection logic
            elif waited > 5 or gap <= 2:
                # Timeout or danger — give up waiting
                STATE["mine_wait"] = False
                STATE["mine_invested"] = None
            elif gap > 2:
                # Stay put, wait for miner
                actions[uid] = "IDLE"
                reserved.add((c, r))
                return

        if my_mines_nearby and gap > 2:
            # Stay and collect energy — move onto mine cell if not already
            mc2, mr2 = my_mines_nearby[0]
            if (mc2, mr2) == (c, r):
                actions[uid] = "IDLE"
                reserved.add((c, r))
                return
            # Try to step onto the mine (allow_danger=True since mine is critical
            # and (mc2,mr2) is rarely in enemy zone — but allow risk if it is).
            for d in ("NORTH", "EAST", "WEST", "SOUTH"):
                dc2, dr2, _ = DIRS[d]
                if (c + dc2, r + dr2) == (mc2, mr2):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                                            danger=enemy_danger, allow_danger=True, hard_block=enemy_hard_block):
                        return
            # Can't reach mine cell, just idle
            actions[uid] = "IDLE"
            reserved.add((c, r))
            return

        if my_mines_nearby and gap <= 2:
            # Must leave — clear mine_invested
            STATE["mine_invested"] = None

        center = width // 4
        ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]

        # Build BFS goals with mine target fusion
        if stuck >= 1:
            # When stuck, expand goal rows to find lateral detours (r+1..r+4 instead of just r+2)
            north_goals = []
            for rc in range(r + 1, min(r + 5, obs.northBound + 1)):
                for c2 in range(width):
                    if in_bounds(c2, rc, obs, config):
                        north_goals.append((c2, rc))
        else:
            north_goals = [(c2, r + 2) for c2 in range(width) if in_bounds(c2, r + 2, obs, config)]
        goals = north_goals
        if mine_target:
            # Navigate to cell SOUTH of mine node so miner spawns ON the node
            approach = (mine_target[0], mine_target[1] - 1)
            if approach[1] >= r and in_bounds(approach[0], approach[1], obs, config):
                goals = [approach] + goals
            else:
                goals = [mine_target] + goals

        # If our current cell is in enemy reach, IDLE is itself unsafe — we
        # must move, even into another danger cell (anything is better than
        # standing still while enemy walks/jumps onto us).
        must_escape = (c, r) in enemy_danger
        # When stuck, low gap, or about to be hit, allow crushing own units
        crush = (stuck >= 1) or (gap <= 3) or must_escape
        # Allow stepping into enemy danger only when scroll pressure is genuinely
        # high (gap<=3) OR we already sit on a cell the enemy will reach.
        # NOTE: We deliberately do NOT include `stuck` here — stuck-because-of-
        # enemy-avoidance is the desired behaviour. If we counted stuck, the
        # factory would IDLE three turns to dodge a JUMP threat and then walk
        # straight into the JUMP-landing on turn 4 (this is the seed 6344
        # failure mode). When the scroll truly threatens us, gap<=3 takes over.
        panic = (gap <= 3) or must_escape

        # Tier 1: Direct NORTH if no known wall
        if can_go(obs, config, c, r, "NORTH"):
            if factory_try_move(uid, c, r, "NORTH", obs, config, actions, reserved, occupied, my_player,
                                                allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                return

        # Tier 2: BFS to goals (mine target + row+2), wider search when stuck
        bfs_limit = 600 if stuck >= 1 else 300
        step_dir = bfs_first_step((c, r), goals, obs, config, can_go, max_nodes=bfs_limit)
        if step_dir:
            dc2, dr2, _ = DIRS[step_dir]
            if dr2 >= 0:  # NORTH, EAST, or WEST only
                if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                                    allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

        # Tier 2b: Pessimistic BFS when stuck (known-safe routes only)
        if stuck >= 2:
            step_dir = bfs_first_step((c, r), goals, obs, config, can_go_pessimistic, max_nodes=bfs_limit)
            if step_dir:
                dc2, dr2, _ = DIRS[step_dir]
                if dr2 >= 0:
                    if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                                        allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                        return

        # Tier 2c: Lateral escape BFS when stuck — find nearest cell with open north
        if stuck >= 2:
            lateral_goals = []
            for cc in range(width):
                if cc == c:
                    continue
                for rc in range(max(r - 1, obs.southBound), min(r + 3, obs.northBound + 1)):
                    if in_bounds(cc, rc, obs, config) and can_go(obs, config, cc, rc, "NORTH"):
                        lateral_goals.append((cc, rc))
            if lateral_goals:
                step_dir = bfs_first_step((c, r), lateral_goals, obs, config, can_go, max_nodes=600)
                if step_dir:
                    dc2, dr2, _ = DIRS[step_dir]
                    if dr2 >= 0:
                        if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                            allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                            return

        # Tier 3: Forced lateral
        for d in ew:
            if can_go(obs, config, c, r, d):
                if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                                    allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

        # Tier 4: Diagonal
        for d in ew:
            if can_go(obs, config, c, r, d):
                dc2, dr2, _ = DIRS[d]
                side = (c + dc2, r)
                if can_go(obs, config, side[0], side[1], "NORTH"):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                                        allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                        return

        # Tier 5: BFS allowing south (stuck >= 3). Stuck recovery without
        # crashing into an enemy factory: keep allow_danger=panic so we still
        # IDLE in preference to a guaranteed factory collision when scroll
        # pressure is low.
        if stuck >= 3:
            step_dir = bfs_first_step((c, r), goals, obs, config, can_go)
            if step_dir:
                if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                    allow_crush=True, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

        # Tier 6: SOUTH as last resort
        if stuck >= 4 and gap >= 3:
            if can_go(obs, config, c, r, "SOUTH"):
                if factory_try_move(uid, c, r, "SOUTH", obs, config, actions, reserved, occupied, my_player,
                                    allow_crush=True, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

    # ── BUILD (during move cooldown) ──
    if move_cd != 0 and build_cd == 0 and gap >= 2:
        spawn_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if spawn_ok:
            spawn = (c, r + 1)
            if not friendly_at(occupied, spawn, my_player):
                # Check if we have an existing mine nearby to collect from
                my_mines_nearby_build = []
                for mk, mv in getattr(obs, "mines", {}).items():
                    mc2, mr2 = parse_key(mk)
                    if mv[2] == my_player and abs(mc2 - c) + abs(mr2 - r) <= 1:
                        my_mines_nearby_build.append((mc2, mr2))

                if my_mines_nearby_build and gap > 2:
                    # Keep collecting energy
                    actions[uid] = "IDLE"
                    reserved.add((c, r))
                    return

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

                # Build Worker: normal early, rebuild mid/late, multiple allowed after step 400
                max_workers = 2 if turn > 400 else 1
                if worker_count < max_workers:
                    can_build = (energy >= 500 and (turn < 150 or energy >= 700))
                    if not can_build and turn >= 100 and energy >= 400:
                        can_build = True
                    if can_build:
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
