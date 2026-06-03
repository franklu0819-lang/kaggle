"""v11 agent: v10 + NN-guided BFS neighbor ordering.

Changes from v10:
1. Small NN (110->32->4) predicts direction scores for BFS neighbor ordering
2. NN replaces hardcoded ("NORTH","EAST","WEST","SOUTH") in BFS calls
3. Falls back to v10 behavior when no weights loaded
"""
from collections import deque
import numpy as np

# ─── NN Weights (default = identity ordering N,E,W,S) ──────────────────
_NN_LOADED = False
_NN_W1 = np.zeros((32, 110), dtype=np.float32)
_NN_b1 = np.array([3.,1.,.5,0.] + [0.]*28, dtype=np.float32)  # bias N>E>W>S
_NN_W2 = np.zeros((4, 32), dtype=np.float32)
for _i in range(min(4, 32)):
    _NN_W2[_i, _i] = 1.0
_NN_b2 = np.zeros(4, dtype=np.float32)

_DIR_NAMES = ("NORTH", "EAST", "WEST", "SOUTH")


def _nn_load_weights(w1, b1, w2, b2):
    global _NN_W1, _NN_b1, _NN_W2, _NN_b2, _NN_LOADED
    _NN_W1, _NN_b1 = w1, b1
    _NN_W2, _NN_b2 = w2, b2
    _NN_LOADED = True


def extract_bfs_features(c, r, obs, config):
    """Extract 110-dim feature vector for BFS ordering NN."""
    w = config.width
    gap = r - obs.southBound
    stuck = STATE["factory_stuck"]
    turn = STATE["turn"]
    mine_target = STATE.get("mine_invested")

    # 5x5x4 wall grid (100 dims)
    grid = np.zeros(100, dtype=np.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nc, nr2 = c + dc, r + dr
            idx = (nr2 - obs.southBound) * w + nc
            if 0 <= nc < w and obs.southBound <= nr2 <= obs.northBound and 0 <= idx < len(obs.walls):
                v = obs.walls[idx]
                if v != -1:
                    pos = (dr + 2) * 20 + (dc + 2) * 4
                    grid[pos] = float(bool(v & BIT_N))
                    grid[pos+1] = float(bool(v & BIT_E))
                    grid[pos+2] = float(bool(v & BIT_W))
                    grid[pos+3] = float(bool(v & BIT_S))

    has_mt = float(mine_target is not None)
    mt_dc = (mine_target[0] - c) / 10.0 if mine_target else 0.0
    scalars = np.array([
        c / 19.0, gap / 20.0, stuck / 10.0, turn / 500.0,
        float(can_go(obs, config, c, r, "NORTH")),
        float(can_go(obs, config, c, r, "EAST")),
        float(can_go(obs, config, c, r, "WEST")),
        float(can_go(obs, config, c, r, "SOUTH")),
        has_mt, mt_dc,
    ], dtype=np.float32)
    return np.concatenate([grid, scalars])


def nn_predict(features):
    x = features @ _NN_W1.T + _NN_b1
    x = np.maximum(0, x)
    return x @ _NN_W2.T + _NN_b2


def get_bfs_order(c, r, obs, config):
    if not _NN_LOADED:
        return None
    features = extract_bfs_features(c, r, obs, config)
    scores = nn_predict(features)
    indices = np.argsort(-scores)
    return tuple(_DIR_NAMES[i] for i in indices)

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


def bfs_first_step(start, goals, obs, config, passable_fn, max_nodes=500, dir_order=None):
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return None
    q = deque([(start, None)])
    visited = {start}
    best_fd, best_dist = None, 999999
    order = dir_order or ("NORTH", "EAST", "WEST", "SOUTH")
    while q:
        cur, first_d = q.popleft()
        dist = min(abs(cur[0] - g[0]) + abs(cur[1] - g[1]) for g in goals)
        if dist < best_dist:
            best_dist = dist
            best_fd = first_d
        for d in order:
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


def bfs_distance(start, goal, obs, config, passable_fn, max_nodes=500, dir_order=None):
    if start == goal:
        return 0
    q = deque([(start, 0)])
    visited = {start}
    order = dir_order or ("NORTH", "EAST", "WEST", "SOUTH")
    while q:
        cur, dist = q.popleft()
        for d in order:
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
    mc, mr = mine_node
    approach = (mc, mr - 1)
    if approach[1] < factory_r or not in_bounds(mc, approach[1], obs, config):
        return 0
    row_gain = approach[1] - factory_r
    dist = bfs_distance((factory_c, factory_r), approach, obs, config, can_go)
    if dist is None:
        return 0
    turns_to_reach = dist * 2
    start_int = getattr(config, "scrollStartInterval", 4)
    end_int = getattr(config, "scrollEndInterval", 1)
    ramp_steps = getattr(config, "scrollRampSteps", 400)
    progress = min(1.0, step / ramp_steps)
    scroll_interval = max(float(end_int), start_int - (start_int - end_int) * progress)
    gap_at_arrival = gap + row_gain - turns_to_reach / scroll_interval
    stay_turns = gap_at_arrival - 2
    effective_stay = max(0, stay_turns - 3)
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
    hard_block = set()
    danger = set()
    for uid, d in obs.robots.items():
        if d[4] == my_player or d[0] != TYPE_FACTORY:
            continue
        ec, er = d[1], d[2]
        emcd = d[5] if len(d) > 5 else 0
        ejcd = d[6] if len(d) > 6 else 0
        hard_block.add((ec, er))
        danger.add((ec, er))
        if emcd == 0:
            for d_str in ("NORTH", "EAST", "WEST", "SOUTH"):
                if can_go(obs, config, ec, er, d_str):
                    dc, dr, _ = DIRS[d_str]
                    danger.add((ec + dc, er + dr))
        if ejcd == 0:
            for jdc, jdr in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                lc, lr = ec + jdc, er + jdr
                if in_bounds(lc, lr, obs, config):
                    danger.add((lc, lr))
    return hard_block, danger


def factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                     allow_crush=False, danger=None, allow_danger=False, hard_block=None):
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

    enemy_hard_block, enemy_danger = get_enemy_factory_threat(obs, config, my_player)

    scout_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_SCOUT)
    worker_count = sum(1 for d in obs.robots.values()
                       if d[4] == my_player and d[0] == TYPE_WORKER)
    miner_count = sum(1 for d in obs.robots.values()
                      if d[4] == my_player and d[0] == TYPE_MINER)
    my_mines = sum(1 for k, v in getattr(obs, "mines", {}).items() if v[2] == my_player)

    # ── Mine target selection ──
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

    # ── JUMP (aggressive: always JUMP when available) ──
    if jump_cd == 0 and turn > 2 and not STATE.get("mine_wait", False):
        # Pre-compute danger escape for lateral jump decision
        move_targets = []
        for d_str in ("NORTH", "EAST", "WEST", "SOUTH"):
            if can_go(obs, config, c, r, d_str):
                dc_t, dr_t, _ = DIRS[d_str]
                move_targets.append((c + dc_t, r + dr_t))
        danger_escape = bool(move_targets) and all(t in enemy_danger for t in move_targets)

        allow_danger_jump = (gap <= 3)
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

        # Lateral jumps: emergency (gap≤3) or danger escape
        if gap <= 3 or danger_escape:
            for jd, (jdc, jdr) in (("JUMP_EAST", (2, 0)), ("JUMP_WEST", (-2, 0))):
                lc, lr2 = c + jdc, r + jdr
                if not in_bounds(lc, lr2, obs, config):
                    continue
                if (lc, lr2) in enemy_hard_block:
                    continue
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

    # ── Mine handling (MOVE/IDLE, requires move_cd == 0 for MOVE) ──
    if move_cd == 0:
        my_mines_nearby = []
        for mk, mv in getattr(obs, "mines", {}).items():
            mc2, mr2 = parse_key(mk)
            if mv[2] == my_player and abs(mc2 - c) + abs(mr2 - r) <= 1:
                my_mines_nearby.append((mc2, mr2))

        if STATE["mine_wait"]:
            mine_exists_nearby = any(
                mv[2] == my_player and abs(parse_key(mk)[0] - c) + abs(parse_key(mk)[1] - r) <= 1
                for mk, mv in getattr(obs, "mines", {}).items()
            )
            waited = turn - STATE["mine_wait_since"]
            if mine_exists_nearby:
                STATE["mine_wait"] = False
            elif waited > 5 or gap <= 2:
                STATE["mine_wait"] = False
                STATE["mine_invested"] = None
            elif gap > 2:
                actions[uid] = "IDLE"
                reserved.add((c, r))
                return

        if my_mines_nearby and gap > 2:
            mc2, mr2 = my_mines_nearby[0]
            if (mc2, mr2) == (c, r):
                actions[uid] = "IDLE"
                reserved.add((c, r))
                return
            for d in ("NORTH", "EAST", "WEST", "SOUTH"):
                dc2, dr2, _ = DIRS[d]
                if (c + dc2, r + dr2) == (mc2, mr2):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                                            danger=enemy_danger, allow_danger=True, hard_block=enemy_hard_block):
                        return
            actions[uid] = "IDLE"
            reserved.add((c, r))
            return

        if my_mines_nearby and gap <= 2:
            STATE["mine_invested"] = None

    # ── Navigation ──
    dir_order = get_bfs_order(c, r, obs, config)
    center = width // 4
    ew = ["EAST", "WEST"] if c <= center else ["WEST", "EAST"]

    north_goals = [(c2, r + 2) for c2 in range(width) if in_bounds(c2, r + 2, obs, config)]
    goals = north_goals
    if mine_target:
        approach = (mine_target[0], mine_target[1] - 1)
        if approach[1] >= r and in_bounds(approach[0], approach[1], obs, config):
            goals = [approach] + goals
        else:
            goals = [mine_target] + goals

    must_escape = (c, r) in enemy_danger
    crush = (stuck >= 1) or (gap <= 3) or must_escape
    panic = (gap <= 3) or must_escape
    bfs_limit = 500

    # Tier 1: Direct NORTH (MOVE)
    if move_cd == 0 and can_go(obs, config, c, r, "NORTH"):
        if factory_try_move(uid, c, r, "NORTH", obs, config, actions, reserved, occupied, my_player,
                            allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
            return

    # Tier 1b: Lateral escape BFS (MOVE)
    if move_cd == 0:
        lateral_goals = []
        for cc in range(width):
            if cc == c:
                continue
            for rc in range(max(r - 1, obs.southBound), min(r + 3, obs.northBound + 1)):
                if in_bounds(cc, rc, obs, config) and can_go(obs, config, cc, rc, "NORTH"):
                    lateral_goals.append((cc, rc))
        if lateral_goals:
            step_dir = bfs_first_step((c, r), lateral_goals, obs, config, can_go, max_nodes=600, dir_order=dir_order)
            if step_dir:
                dc2, dr2, _ = DIRS[step_dir]
                if dr2 >= 0:
                    if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                        allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                        return

    # Tier 2: BFS to goals (MOVE)
    if move_cd == 0:
        step_dir = bfs_first_step((c, r), goals, obs, config, can_go, max_nodes=bfs_limit, dir_order=dir_order)
        if step_dir:
            dc2, dr2, _ = DIRS[step_dir]
            if dr2 >= 0:
                if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                    allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

    # Tier 2b: Pessimistic BFS (MOVE, stuck >= 2)
    if move_cd == 0 and stuck >= 2:
        step_dir = bfs_first_step((c, r), goals, obs, config, can_go_pessimistic, max_nodes=bfs_limit, dir_order=dir_order)
        if step_dir:
            dc2, dr2, _ = DIRS[step_dir]
            if dr2 >= 0:
                if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                    allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

    # Tier 3: Forced lateral (MOVE)
    if move_cd == 0:
        for d in ew:
            if can_go(obs, config, c, r, d):
                if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                    allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                    return

    # Tier 4: Diagonal (MOVE)
    if move_cd == 0:
        for d in ew:
            if can_go(obs, config, c, r, d):
                dc2, dr2, _ = DIRS[d]
                side = (c + dc2, r)
                if can_go(obs, config, side[0], side[1], "NORTH"):
                    if factory_try_move(uid, c, r, d, obs, config, actions, reserved, occupied, my_player,
                                        allow_crush=crush, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                        return

    # Tier 5: BFS allowing south (MOVE, stuck >= 3)
    if move_cd == 0 and stuck >= 3:
        step_dir = bfs_first_step((c, r), goals, obs, config, can_go, dir_order=dir_order)
        if step_dir:
            if factory_try_move(uid, c, r, step_dir, obs, config, actions, reserved, occupied, my_player,
                                allow_crush=True, danger=enemy_danger, allow_danger=panic, hard_block=enemy_hard_block):
                return

    # Tier 6: SOUTH (MOVE, stuck >= 4)
    if move_cd == 0 and stuck >= 4 and gap >= 3:
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
                my_mines_nearby_build = []
                for mk, mv in getattr(obs, "mines", {}).items():
                    mc2, mr2 = parse_key(mk)
                    if mv[2] == my_player and abs(mc2 - c) + abs(mr2 - r) <= 1:
                        my_mines_nearby_build.append((mc2, mr2))

                if my_mines_nearby_build and gap > 2:
                    actions[uid] = "IDLE"
                    reserved.add((c, r))
                    return

                if mine_target and energy >= 400:
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

    if gap <= 1 and factory_pos and energy > 5:
        fc, fr = factory_pos
        for d, (dc, dr, _) in [("NORTH", DIRS["NORTH"]), ("SOUTH", DIRS["SOUTH"]),
                                 ("EAST", DIRS["EAST"]), ("WEST", DIRS["WEST"])]:
            if (c + dc, r + dr) == (fc, fr) and can_go(obs, config, c, r, d):
                actions[uid] = f"TRANSFER_{d}"
                reserved.add((c, r))
                return

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


# ─── Scout ──────────────────────────────────────────────────────────────

def scout_action(uid, data, obs, config, actions, reserved, occupied, my_player):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    gap = r - obs.southBound

    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return

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

    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_SCOUT:
            scout_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_WORKER:
            worker_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if uid not in actions and data[4] == my_player and data[0] == TYPE_MINER:
            miner_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    for uid, data in obs.robots.items():
        if data[4] == my_player and data[0] == TYPE_FACTORY:
            factory_action(uid, data, obs, config, actions, reserved, occupied, my_player)

    return actions
