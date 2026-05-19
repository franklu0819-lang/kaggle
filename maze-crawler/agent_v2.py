from collections import deque
from math import inf
import random

# Shared state across turns inside Kaggle's notebook runtime.
STATE = {
    "turn": 0,
    "walls": {},          # (col, row) -> wall bitfield
    "nodes": set(),       # discovered mining nodes
    "mines": {},          # (col, row) -> [energy, maxEnergy, owner]
    "enemy_factory": None,
    "my_factory": None,
    "enemy_seen": {},     # uid -> last visible enemy data
    "factory_stuck": 0,   # consecutive turns factory hasn't moved
    "factory_last_pos": None,
}

TYPE_FACTORY = 0
TYPE_SCOUT = 1
TYPE_WORKER = 2
TYPE_MINER = 3

BIT_N, BIT_E, BIT_S, BIT_W = 1, 2, 4, 8

DIRS = {
    "NORTH": (0, 1, BIT_N),
    "EAST": (1, 0, BIT_E),
    "SOUTH": (0, -1, BIT_S),
    "WEST": (-1, 0, BIT_W),
}
DIR_ORDER = ("NORTH", "EAST", "WEST", "SOUTH")
# North-first ordering with south heavily deprioritized for factory
DIR_ORDER_FACTORY = ("NORTH", "EAST", "WEST", "SOUTH")
OPPOSITE_BIT = {
    "NORTH": BIT_S,
    "EAST": BIT_W,
    "SOUTH": BIT_N,
    "WEST": BIT_E,
}
MOVE_ACTIONS = set(DIRS)

# Northward preference weights for BFS
DIR_NORTH_WEIGHT = {"NORTH": 0, "EAST": 1, "WEST": 1, "SOUTH": 4}


def parse_key(key):
    c, r = key.split(",")
    return int(c), int(r)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(c, r, obs, config):
    return 0 <= c < config.width and obs.southBound <= r <= obs.northBound


def visible_range_rows(obs):
    return range(obs.southBound, obs.northBound + 1)


def nearest_point(start, points):
    if not points:
        return None, inf
    best, best_d = None, inf
    for p in points:
        d = manhattan(start, p)
        if d < best_d:
            best, best_d = p, d
    return best, best_d


def scroll_interval(step, config):
    """Predict the current scroll interval based on step count."""
    if step >= config.scrollRampSteps:
        return config.scrollEndInterval
    progress = step / max(1, config.scrollRampSteps)
    interval = config.scrollStartInterval - (
        config.scrollStartInterval - config.scrollEndInterval
    ) * progress
    return max(config.scrollEndInterval, round(interval))


def update_memory(obs, config):
    STATE["turn"] += 1

    width = config.width
    for r in visible_range_rows(obs):
        base = (r - obs.southBound) * width
        if base < 0 or base >= len(obs.walls):
            continue
        for c in range(width):
            idx = base + c
            if idx >= len(obs.walls):
                break
            w = obs.walls[idx]
            if w != -1:
                STATE["walls"][(c, r)] = int(w)

    for key in getattr(obs, "miningNodes", {}) or {}:
        STATE["nodes"].add(parse_key(key))

    for key, val in getattr(obs, "mines", {}).items():
        STATE["mines"][parse_key(key)] = list(val)

    STATE["enemy_seen"].clear()
    for uid, data in obs.robots.items():
        if data[4] != obs.player:
            STATE["enemy_seen"][uid] = tuple(data)

    my_factories = []
    for uid, data in obs.robots.items():
        if data[4] == obs.player and data[0] == TYPE_FACTORY:
            my_factories.append((uid, data))
    if my_factories:
        uid, data = my_factories[0]
        STATE["my_factory"] = (uid, data[1], data[2], data[3])
        # Track factory "no north progress" counter
        # (resets only on northward movement, not on south backtrack)
        pos = (data[1], data[2])
        last_pos = STATE.get("factory_last_pos")
        if last_pos is not None:
            if data[2] > last_pos[1]:  # moved north = progress
                STATE["factory_stuck"] = 0
            else:  # same pos, or moved south/east/west
                STATE["factory_stuck"] = STATE.get("factory_stuck", 0) + 1
        else:
            STATE["factory_stuck"] = 0
        STATE["factory_last_pos"] = pos

    visible_enemy_factory = None
    for uid, data in obs.robots.items():
        if data[4] != obs.player and data[0] == TYPE_FACTORY:
            visible_enemy_factory = (uid, data[1], data[2], data[3])
            break
    if visible_enemy_factory:
        STATE["enemy_factory"] = visible_enemy_factory


def wall_bits_at(c, r):
    return STATE["walls"].get((c, r))


def blocked(c, r, direction, obs, config):
    dc, dr, bit = DIRS[direction]
    nc, nr = c + dc, r + dr
    if not in_bounds(nc, nr, obs, config):
        return True

    here = wall_bits_at(c, r)
    there = wall_bits_at(nc, nr)

    # If we haven't seen either cell, assume fully walled (maze starts fully walled)
    if here is None or there is None:
        return True

    if here & bit:
        return True
    if there & OPPOSITE_BIT[direction]:
        return True

    return False


def has_north_wall(c, r):
    """Check if there is a known wall to the north of (c, r)."""
    w = wall_bits_at(c, r)
    if w is not None and (w & BIT_N):
        return True
    w_above = wall_bits_at(c, r + 1)
    if w_above is not None and (w_above & BIT_S):
        return True
    return False


def bfs_first_step(start, goal, obs, config, north_bias=False):
    if start == goal:
        return None

    q = deque()
    prev = {start: None}
    prev_dir = {start: None}
    visited = {start}

    best = start
    best_score = manhattan(start, goal)

    # Use priority ordering for initial queue
    init_dirs = DIR_ORDER
    for d in init_dirs:
        dc, dr, _ = DIRS[d]
        nxt = (start[0] + dc, start[1] + dr)
        if nxt in visited:
            continue
        if not in_bounds(nxt[0], nxt[1], obs, config):
            continue
        if blocked(start[0], start[1], d, obs, config):
            continue
        visited.add(nxt)
        prev[nxt] = start
        prev_dir[nxt] = d
        q.append(nxt)

    while q:
        cur = q.popleft()
        score = manhattan(cur, goal)
        if north_bias:
            # Penalize southward positions heavily
            score += max(0, start[1] - cur[1]) * 3
        if score < best_score:
            best = cur
            best_score = score

        if cur == goal:
            best = cur
            break

        for d in DIR_ORDER:
            dc, dr, _ = DIRS[d]
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt in visited:
                continue
            if not in_bounds(nxt[0], nxt[1], obs, config):
                continue
            if blocked(cur[0], cur[1], d, obs, config):
                continue
            visited.add(nxt)
            prev[nxt] = cur
            prev_dir[nxt] = d
            q.append(nxt)

    target = goal if goal in visited else best
    if target == start:
        return None

    cur = target
    while prev[cur] != start:
        cur = prev[cur]
        if cur is None:
            return None
    return prev_dir[cur] if prev[cur] == start else None


def known_blocked(c, r, direction, obs, config):
    """Check if movement is blocked by a KNOWN wall. Unknown cells = passable."""
    dc, dr, bit = DIRS[direction]
    nc, nr = c + dc, r + dr
    if not in_bounds(nc, nr, obs, config):
        return True
    here = wall_bits_at(c, r)
    if here is not None and (here & bit):
        return True
    there = wall_bits_at(nc, nr)
    if there is not None and (there & OPPOSITE_BIT[direction]):
        return True
    return False


def target_to_step(start, target, obs, config, north_bias=False):
    # Tier 1: BFS through known cells only
    step = bfs_first_step(start, target, obs, config, north_bias=north_bias)
    if step is not None:
        return step

    # Tier 2: Greedy move using known walls only (allows exploring unknown territory)
    candidates = []
    for d in DIR_ORDER:
        dc, dr, _ = DIRS[d]
        nxt = (start[0] + dc, start[1] + dr)
        if not in_bounds(nxt[0], nxt[1], obs, config):
            continue
        if known_blocked(start[0], start[1], d, obs, config):
            continue
        dist = manhattan(nxt, target)
        if north_bias:
            dist += max(0, start[1] - nxt[1]) * 3
        candidates.append((dist, d))
    if not candidates:
        # Tier 3: Even if known wall exists, try to go (may be outdated info)
        # Just avoid going out of bounds
        for d in DIR_ORDER:
            dc, dr, _ = DIRS[d]
            nxt = (start[0] + dc, start[1] + dr)
            if in_bounds(nxt[0], nxt[1], obs, config):
                return d
        return "IDLE"
    candidates.sort()
    return candidates[0][1]


def visible_enemy_targets(obs):
    out = []
    for uid, data in obs.robots.items():
        if data[4] != obs.player:
            out.append((uid, data))
    return out


def team_energy(obs):
    return sum(data[3] for data in obs.robots.values() if data[4] == obs.player)


def count_units(obs, robot_type=None):
    if robot_type is None:
        return sum(1 for data in obs.robots.values() if data[4] == obs.player)
    return sum(1 for data in obs.robots.values()
               if data[4] == obs.player and data[0] == robot_type)


def mirrored_enemy_guess(obs, config):
    if STATE["enemy_factory"] is not None:
        _, c, r, _ = STATE["enemy_factory"]
        return (c, r)
    if STATE["my_factory"] is None:
        return None
    _, c, r, _ = STATE["my_factory"]
    return (config.width - 1 - c, r)


def nearby_visible_crystals(obs):
    crystals = []
    for key, energy in (getattr(obs, "crystals", {}) or {}).items():
        crystals.append((parse_key(key), energy))
    return crystals


def visible_mining_nodes(obs):
    return [parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {})]


def maybe_transfer(uid, data, obs, config, actions, reserved):
    rtype, c, r, energy = data[0], data[1], data[2], data[3]
    if energy <= 1:
        return False

    best_target = None
    best_score = -inf
    for other_uid, other in obs.robots.items():
        if other_uid == uid or other[4] != obs.player:
            continue
        oc, orow, oenergy = other[1], other[2], other[3]
        if abs(oc - c) + abs(orow - r) != 1:
            continue
        score = 0.0
        if other[0] == TYPE_FACTORY:
            score += 1000.0
        elif other[0] == TYPE_MINER:
            score += 250.0
        elif other[0] == TYPE_WORKER:
            score += 150.0
        else:
            score += 50.0
        score += max(0, 200 - oenergy)
        score += min(50, energy)
        if score > best_score:
            best_score = score
            best_target = (other_uid, other)

    if best_target is None:
        return False

    _, other = best_target
    oc, orow, oenergy = other[1], other[2], other[3]

    # Transfer minimum of own energy and what target can accept
    max_e_target = {TYPE_FACTORY: float("inf"), TYPE_SCOUT: 100,
                    TYPE_WORKER: 300, TYPE_MINER: 500}.get(other[0], 0)
    space = max_e_target - oenergy
    if space <= 0:
        return False

    transfer_amount = min(energy, max(1, int(space) if space != float("inf") else energy))
    # Keep at least 1 energy to survive next turn's drain
    if energy - transfer_amount < 2:
        transfer_amount = max(0, energy - 2)

    if transfer_amount <= 0:
        return False

    if other[0] == TYPE_FACTORY and transfer_amount >= 30:
        direction = None
        if oc == c + 1:
            direction = "TRANSFER_EAST"
        elif oc == c - 1:
            direction = "TRANSFER_WEST"
        elif orow == r + 1:
            direction = "TRANSFER_NORTH"
        elif orow == r - 1:
            direction = "TRANSFER_SOUTH"
        if direction:
            actions[uid] = direction
            reserved.add((c, r))
            return True

    if rtype in (TYPE_WORKER, TYPE_MINER) and transfer_amount >= 30:
        direction = None
        if oc == c + 1:
            direction = "TRANSFER_EAST"
        elif oc == c - 1:
            direction = "TRANSFER_WEST"
        elif orow == r + 1:
            direction = "TRANSFER_NORTH"
        elif orow == r - 1:
            direction = "TRANSFER_SOUTH"
        if direction:
            actions[uid] = direction
            reserved.add((c, r))
            return True

    return False


strength_rank = {
    TYPE_FACTORY: 4,
    TYPE_MINER: 3,
    TYPE_WORKER: 2,
    TYPE_SCOUT: 1,
}

max_energy = {
    TYPE_SCOUT: 100,
    TYPE_WORKER: 300,
    TYPE_MINER: 500,
}


def current_occupants(obs):
    occ = {}
    for uid, data in obs.robots.items():
        cell = (data[1], data[2])
        occ.setdefault(cell, []).append((uid, data))
    return occ


def best_attack_step(uid, data, obs, config, occupied):
    if data[5] != 0:
        return None

    c, r = data[1], data[2]
    my_strength = strength_rank[data[0]]

    best = None
    best_score = -inf

    for d in DIR_ORDER:
        dc, dr, _ = DIRS[d]
        nxt = (c + dc, r + dr)
        if not in_bounds(nxt[0], nxt[1], obs, config):
            continue
        if blocked(c, r, d, obs, config):
            continue

        occupants = occupied.get(nxt, [])
        allies = [o for o in occupants if o[1][4] == obs.player]
        enemies = [o for o in occupants if o[1][4] != obs.player]
        if allies or not enemies:
            continue

        enemy_strength = max(strength_rank[o[1][0]] for o in enemies)
        if my_strength <= enemy_strength:
            continue

        score = 10.0 * (my_strength - enemy_strength)
        score += 0.5 * len(enemies)
        if any(o[1][0] == TYPE_FACTORY for o in enemies):
            score += 100.0
        if any(o[1][0] == TYPE_MINER for o in enemies):
            score += 20.0
        if score > best_score:
            best_score = score
            best = d

    return best


def on_friendly_mine(uid, data, obs):
    cell = (data[1], data[2])
    mine = STATE["mines"].get(cell)
    return bool(mine and mine[2] == obs.player)


def choose_scout_target(uid, data, obs, config):
    c, r = data[1], data[2]

    # Top priority: nearby crystals (energy income)
    crystals = nearby_visible_crystals(obs)
    if crystals:
        best = None
        best_score = -inf
        for cell, energy in crystals:
            dist = manhattan((c, r), cell)
            if dist == 0:
                continue
            score = energy / max(1, dist) * 2
            # Prefer north crystals to stay safe from scroll
            score += 0.3 * (cell[1] - r)
            if score > best_score:
                best_score = score
                best = cell
        if best is not None:
            return best

    # Follow factory and explore slightly ahead
    if STATE["my_factory"] is not None:
        _, fc, fr, _ = STATE["my_factory"]
        # Explore 4-6 cells ahead of factory
        target = (fc, min(obs.northBound, fr + 6))
        return target

    # Fallback: explore north
    target_row = min(obs.northBound, r + 8)
    half = config.width // 2
    if c < half:
        target_col = max(0, min(half - 2, c + (2 if STATE["turn"] % 20 < 10 else -2)))
    else:
        target_col = max(half + 1, min(config.width - 1, c + (2 if STATE["turn"] % 20 < 10 else -2)))

    return (target_col, target_row)


def choose_worker_target(uid, data, obs, config):
    c, r = data[1], data[2]

    # Follow factory closely and clear walls in its path
    if STATE["my_factory"] is not None:
        _, fc, fr, _ = STATE["my_factory"]
        # Check if there's a wall directly north of factory
        if has_north_wall(fc, fr):
            # Go to factory's north neighbor cell to remove that wall
            return (fc, fr + 1)
        # Stay just north of factory to clear upcoming walls
        target = (fc, min(obs.northBound, fr + 2))
        return target

    return (c, min(obs.northBound, r + 6))


def choose_miner_target(uid, data, obs, config):
    c, r = data[1], data[2]
    nodes = [p for p in STATE["nodes"] if p not in STATE["mines"]]
    if nodes:
        target, _ = nearest_point((c, r), nodes)
        if target is not None:
            return target
    # Follow factory north
    if STATE["my_factory"] is not None:
        _, fc, fr, _ = STATE["my_factory"]
        return (fc, min(obs.northBound, fr + 2))
    return (c, min(obs.northBound, r + 6))


def remove_direction_if_blocked(uid, data, obs, config, actions, reserved):
    """Remove walls blocking factory's northward path."""
    c, r = data[1], data[2]
    build_cd = data[7] if len(data) > 7 else 0
    if build_cd != 0 or data[3] < getattr(config, "wallRemoveCost", 100):
        return False

    # Priority: remove walls that block the factory's path north
    if STATE["my_factory"] is not None:
        _, fc, fr, _ = STATE["my_factory"]
        # If we're adjacent to factory, remove wall in factory's desired direction
        if abs(c - fc) + abs(r - fr) <= 2:
            # Check if factory's north is blocked
            factory_north_blocked = has_north_wall(fc, fr)
            if factory_north_blocked and fr + 1 == r and fc == c:
                # We're at factory's north cell, remove north wall here
                w = wall_bits_at(c, r)
                if w is not None and (w & BIT_N):
                    actions[uid] = "REMOVE_NORTH"
                    reserved.add((c, r))
                    return True

    # General: remove walls blocking north movement
    for d, bit, opp_bit in [("NORTH", BIT_N, BIT_S), ("EAST", BIT_E, BIT_W),
                             ("WEST", BIT_W, BIT_E)]:
        dc, dr, _ = DIRS[d]
        nc, nr = c + dc, r + dr
        if not in_bounds(nc, nr, obs, config):
            continue

        w_here = wall_bits_at(c, r)
        w_there = wall_bits_at(nc, nr)
        if (w_here is not None and (w_here & bit)) or \
           (w_there is not None and (w_there & opp_bit)):
            actions[uid] = f"REMOVE_{d}"
            reserved.add((c, r))
            return True

    return False


def factory_move_direction(c, r, obs, config, occupied, reserved):
    """Factory movement: NORTH > BFS detour > E/W unknown > E/W any > SOUTH."""
    start = (c, r)
    stuck = STATE.get("factory_stuck", 0)
    safety = r - obs.southBound

    # 1. Direct NORTH if no known wall (fast path)
    if not known_blocked(c, r, "NORTH", obs, config):
        nxt = (c, r + 1)
        if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
            occ = occupied.get(nxt, [])
            if not any(o[1][4] == obs.player for o in occ):
                return "NORTH"

    # 2. BFS through known cells for north-only detour (no south)
    target = (c, min(obs.northBound, r + 25))
    bfs_step = bfs_first_step(start, target, obs, config, north_bias=True)
    if bfs_step and bfs_step in MOVE_ACTIONS:
        dc, dr, _ = DIRS[bfs_step]
        nxt = (c + dc, r + dr)
        if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
            occ = occupied.get(nxt, [])
            if not any(o[1][4] == obs.player for o in occ):
                if dr >= 0:  # NORTH, EAST, or WEST only
                    return bfs_step

    # 3. Diagonal exploration: try NORTHEAST/NORTHWEST (go sideways then up)
    # This helps when direct north is blocked but there's a path 2 cells away
    for d in ["EAST", "WEST"]:
        if not known_blocked(c, r, d, obs, config):
            dc, dr, _ = DIRS[d]
            side = (c + dc, r)
            if not in_bounds(side[0], side[1], obs, config):
                continue
            # Check if we can go north from the side cell
            if not known_blocked(side[0], side[1], "NORTH", obs, config):
                side_north = (side[0], side[1] + 1)
                if in_bounds(side_north[0], side_north[1], obs, config) and side_north not in reserved:
                    occ = occupied.get(side, [])
                    if not any(o[1][4] == obs.player for o in occ):
                        return d  # Move east/west, next turn go north

    # 4. EAST/WEST through unknown territory (explore)
    for d in ["EAST", "WEST"]:
        if not known_blocked(c, r, d, obs, config):
            dc, dr, _ = DIRS[d]
            nxt = (c + dc, r + dr)
            if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
                occ = occupied.get(nxt, [])
                if not any(o[1][4] == obs.player for o in occ):
                    return d

    # 5. BFS south path (only if truly stuck and safety allows)
    if bfs_step and bfs_step == "SOUTH" and stuck >= 4:
        min_safety = 2 if stuck >= 6 else 4
        if safety >= min_safety:
            nxt = (c, r - 1)
            if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
                occ = occupied.get(nxt, [])
                if not any(o[1][4] == obs.player for o in occ):
                    if not known_blocked(c, r, "SOUTH", obs, config):
                        return "SOUTH"

    # 6. Direct south when very stuck
    if stuck >= 6 and safety >= 2 and not known_blocked(c, r, "SOUTH", obs, config):
        nxt = (c, r - 1)
        if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
            occ = occupied.get(nxt, [])
            if not any(o[1][4] == obs.player for o in occ):
                return "SOUTH"

    # 7. Absolute last resort: NORTH anyway (even through wall)
    if in_bounds(c, r + 1, obs, config) and (c, r + 1) not in reserved:
        return "NORTH"

    return None


def decide_factory(uid, data, obs, config, actions, reserved, occupied, rng):
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0

    safety_gap = r - obs.southBound
    scout_count = count_units(obs, TYPE_SCOUT)
    worker_count = count_units(obs, TYPE_WORKER)
    miner_count = count_units(obs, TYPE_MINER)
    spawn = (c, r + 1)

    north_walled = has_north_wall(c, r)
    spawn_clear = (in_bounds(spawn[0], spawn[1], obs, config)
                   and spawn not in occupied
                   and not north_walled)

    # === PRIORITY 1: JUMP when stuck or north is blocked ===
    if jump_cd == 0 and STATE["turn"] > 2 and r + 2 <= obs.northBound:
        north_walled = known_blocked(c, r, "NORTH", obs, config)
        stuck = STATE.get("factory_stuck", 0)
        if north_walled or stuck >= 6:
            actions[uid] = "JUMP_NORTH"
            reserved.add((c, r + 2))
            return True

    # === PRIORITY 2: Move (always move when possible) ===
    if move_cd == 0:
        d = factory_move_direction(c, r, obs, config, occupied, reserved)
        if d:
            dc, dr, _ = DIRS[d]
            nxt = (c + dc, r + dr)
            reserved.add(nxt)
            actions[uid] = d
            return True

    # === PRIORITY 3: Build when factory can't move AND is very safe ===
    if move_cd != 0 and build_cd == 0 and spawn_clear and safety_gap >= 10:
        scout_cost = getattr(config, "scoutCost", 50)
        if scout_count < 1 and energy >= scout_cost + 200:
            actions[uid] = "BUILD_SCOUT"
            reserved.add(spawn)
            return True

    actions[uid] = "IDLE"
    reserved.add((c, r))
    return True


def decide_nonfactory(uid, data, obs, config, actions, reserved, occupied, rng):
    rtype, c, r, energy = data[0], data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    build_cd = data[7] if len(data) > 7 else 0

    # Miner conversion
    if rtype == TYPE_MINER and build_cd == 0 and \
       energy >= getattr(config, "transformCost", 100) + 1:
        if (c, r) in STATE["nodes"]:
            actions[uid] = "TRANSFORM"
            reserved.add((c, r))
            return True

    # Energy transfers
    if maybe_transfer(uid, data, obs, config, actions, reserved):
        return True

    # Worker wall removal
    if rtype == TYPE_WORKER and \
       remove_direction_if_blocked(uid, data, obs, config, actions, reserved):
        return True

    # Attack weaker enemies
    attack_step = best_attack_step(uid, data, obs, config, occupied)
    if attack_step is not None:
        dc, dr, _ = DIRS[attack_step]
        actions[uid] = attack_step
        reserved.add((c + dc, r + dr))
        return True

    # Recharge on friendly mine
    if on_friendly_mine(uid, data, obs) and rtype != TYPE_FACTORY:
        cap = max_energy.get(rtype, 10 ** 9)
        if energy < cap - 5:
            actions[uid] = "IDLE"
            reserved.add((c, r))
            return True

    # Cooldown
    if move_cd != 0:
        actions[uid] = "IDLE"
        reserved.add((c, r))
        return True

    # Keep units moving north to escape scroll
    safety_gap = r - obs.southBound

    # Scouts near factory: move away to unblock factory's north path
    if rtype == TYPE_SCOUT and STATE["my_factory"] is not None:
        _, fc, fr, _ = STATE["my_factory"]
        if c == fc and r == fr + 1:
            # We're directly north of factory, MOVE AWAY
            for d in ["NORTH", "EAST", "WEST"]:
                if not known_blocked(c, r, d, obs, config):
                    dc2, dr2, _ = DIRS[d]
                    nxt = (c + dc2, r + dr2)
                    if in_bounds(nxt[0], nxt[1], obs, config) and nxt not in reserved:
                        occ2 = occupied.get(nxt, [])
                        if not any(o[1][4] == obs.player for o in occ2):
                            actions[uid] = d
                            reserved.add(nxt)
                            return True

    if safety_gap <= 3 and rtype != TYPE_SCOUT:
        # Non-scout units in danger: just go north
        step = target_to_step((c, r), (c, min(obs.northBound, r + 6)), obs, config,
                              north_bias=True)
        if step in MOVE_ACTIONS:
            dc, dr, _ = DIRS[step]
            nxt = (c + dc, r + dr)
            if nxt not in reserved:
                occup = occupied.get(nxt, [])
                if not any(o[1][4] == obs.player for o in occup):
                    actions[uid] = step
                    reserved.add(nxt)
                    return True

    # Role-specific movement
    if rtype == TYPE_SCOUT:
        target = choose_scout_target(uid, data, obs, config)
    elif rtype == TYPE_WORKER:
        target = choose_worker_target(uid, data, obs, config)
    elif rtype == TYPE_MINER:
        target = choose_miner_target(uid, data, obs, config)
    else:
        target = (c, min(obs.northBound, r + 4))

    step = target_to_step((c, r), target, obs, config)
    if step in MOVE_ACTIONS:
        dc, dr, _ = DIRS[step]
        nxt = (c + dc, r + dr)
        if nxt not in reserved:
            occup = occupied.get(nxt, [])
            if not any(o[1][4] == obs.player for o in occup):
                actions[uid] = step
                reserved.add(nxt)
                return True

    actions[uid] = "IDLE"
    reserved.add((c, r))
    return True


def agent(obs, config):
    update_memory(obs, config)

    actions = {}
    reserved = set()
    occupied = current_occupants(obs)
    rng = random.Random(7919 + STATE["turn"] * 17 + obs.player)

    for uid, data in obs.robots.items():
        if data[4] == obs.player and data[0] == TYPE_FACTORY:
            decide_factory(uid, data, obs, config, actions, reserved, occupied, rng)

    others = [
        (uid, data)
        for uid, data in obs.robots.items()
        if data[4] == obs.player and data[0] != TYPE_FACTORY
    ]
    others.sort(key=lambda item: (-strength_rank[item[1][0]], -item[1][3], item[0]))

    for uid, data in others:
        if uid in actions:
            continue
        decide_nonfactory(uid, data, obs, config, actions, reserved, occupied, rng)

    return actions
