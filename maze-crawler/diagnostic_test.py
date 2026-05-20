"""
Diagnostic test: WHY does the v2 agent lose 40% of games against random?
Plays 20 games with detailed factory-level logging every 5 steps.
"""

import sys
import os
import json
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make
from agent import (
    agent as fog_agent, STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    BIT_N, BIT_E, BIT_S, BIT_W, DIRS, known_blocked, has_north_wall
)

# Seeds from user request
SEEDS = [222, 303, 555, 2048] + [42 * 137 + 42 * i for i in range(16)]
SEEDS = SEEDS[:20]  # exactly 20


def reset_state():
    STATE["turn"] = 0
    STATE["walls"] = {}
    STATE["nodes"] = set()
    STATE["mines"] = {}
    STATE["enemy_factory"] = None
    STATE["my_factory"] = None
    STATE["enemy_seen"] = {}
    STATE["factory_stuck"] = 0
    STATE["factory_last_pos"] = None


def run_diagnostic_game(seed):
    """Run one game and collect detailed per-step factory logs."""
    reset_state()

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    env.run([fog_agent, "random"])

    steps = env.steps
    total_steps = len(steps)
    final = steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")

    # Collect factory log for every step
    factory_log = []
    factory_alive_until = total_steps

    for si in range(total_steps):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        my = {k: v for k, v in robots.items() if v[4] == 0}

        factory_data = None
        factory_uid = None
        for uid, data in my.items():
            if data[0] == TYPE_FACTORY:
                factory_data = data
                factory_uid = uid
                break

        if factory_data is None:
            if factory_alive_until == total_steps:
                factory_alive_until = si
            factory_log.append({
                "step": si, "alive": False,
            })
            continue

        c, r = factory_data[1], factory_data[2]
        energy = factory_data[3]
        move_cd = factory_data[5] if len(factory_data) > 5 else 0
        jump_cd = factory_data[6] if len(factory_data) > 6 else 0
        build_cd = factory_data[7] if len(factory_data) > 7 else 0

        southBound = obs.southBound
        northBound = obs.northBound
        safety_gap = r - southBound

        entry = {
            "step": si, "alive": True,
            "pos": (c, r), "energy": energy,
            "southBound": southBound, "northBound": northBound,
            "safety_gap": safety_gap,
            "move_cd": move_cd, "jump_cd": jump_cd, "build_cd": build_cd,
        }
        factory_log.append(entry)

    return {
        "seed": seed, "result": result,
        "our_reward": r0, "opp_reward": r1,
        "total_steps": total_steps,
        "factory_alive_until": factory_alive_until,
        "factory_log": factory_log,
    }


def replay_with_action_logging(seed):
    """
    Re-play the game with patched agent to capture factory actions.
    We intercept the agent's return value at each step.
    """
    reset_state()

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)

    # We need to wrap the agent to capture its actions
    captured_actions = []

    def wrapped_agent(obs, config):
        actions = fog_agent(obs, config)
        captured_actions.append(dict(actions))
        return actions

    env.run([wrapped_agent, "random"])

    steps = env.steps
    total_steps = len(steps)
    final = steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")

    factory_actions = []
    factory_alive_until = total_steps

    for si in range(total_steps):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        my = {k: v for k, v in robots.items() if v[4] == 0}

        factory_uid = None
        factory_data = None
        for uid, data in my.items():
            if data[0] == TYPE_FACTORY:
                factory_uid = uid
                factory_data = data
                break

        if factory_data is None:
            if factory_alive_until == total_steps:
                factory_alive_until = si
            factory_actions.append({
                "step": si, "alive": False, "action": None,
                "pos": None, "safety_gap": None,
                "stuck": None, "move_cd": None, "jump_cd": None,
            })
            continue

        c, r = factory_data[1], factory_data[2]
        action = captured_actions[si].get(factory_uid, "IDLE") if si < len(captured_actions) else "IDLE"
        safety_gap = r - obs.southBound

        factory_actions.append({
            "step": si, "alive": True, "action": action,
            "pos": (c, r), "safety_gap": safety_gap,
            "energy": factory_data[3],
            "stuck": STATE.get("factory_stuck", 0) if si == len(factory_actions) else None,
            "move_cd": factory_data[5] if len(factory_data) > 5 else 0,
            "jump_cd": factory_data[6] if len(factory_data) > 6 else 0,
        })

    return {
        "seed": seed, "result": result,
        "our_reward": r0, "opp_reward": r1,
        "total_steps": total_steps,
        "factory_alive_until": factory_alive_until,
        "factory_actions": factory_actions,
    }


def analyze_loss(game):
    """Deep analysis of a LOSS game's factory actions."""
    actions = game["factory_actions"]
    alive_until = game["factory_alive_until"]

    # Last 10 steps before death
    last_10 = [a for a in actions if a["alive"]][-10:]

    action_counts = Counter(a["action"] for a in actions if a["alive"])
    east_west = action_counts.get("EAST", 0) + action_counts.get("WEST", 0)
    north = action_counts.get("NORTH", 0)

    # Count stuck events (we infer from factory not moving north)
    positions = [(a["step"], a["pos"]) for a in actions if a["alive"]]
    stuck_events = 0
    stuck_ge_4 = 0
    consecutive_same = 0
    prev_pos = None
    for step, pos in positions:
        if prev_pos is not None and pos == prev_pos:
            consecutive_same += 1
            if consecutive_same >= 4:
                stuck_ge_4 += 1
        else:
            if consecutive_same >= 4:
                stuck_events += 1
            consecutive_same = 0
        prev_pos = pos
    if consecutive_same >= 4:
        stuck_events += 1

    jump_triggered = sum(1 for a in actions if a["alive"] and a["action"] == "JUMP_NORTH")

    # Safety gap at death
    death_gap = None
    for a in reversed(actions):
        if a["alive"]:
            death_gap = a["safety_gap"]
            break

    # Was factory killed by scroll? (safety_gap == 0 or 1 at death)
    killed_by_scroll = death_gap is not None and death_gap <= 1

    # How many steps spent at safety_gap <= 2
    danger_steps = sum(1 for a in actions if a["alive"] and a["safety_gap"] is not None and a["safety_gap"] <= 2)

    # North progress rate
    factory_positions = [a["pos"] for a in actions if a["alive"]]
    if factory_positions:
        start_row = factory_positions[0][1]
        max_row = max(p[1] for p in factory_positions)
        north_progress = max_row - start_row
    else:
        north_progress = 0

    return {
        "alive_until": alive_until,
        "death_safety_gap": death_gap,
        "killed_by_scroll": killed_by_scroll,
        "danger_steps": danger_steps,
        "north_progress": north_progress,
        "action_counts": dict(action_counts),
        "east_west_count": east_west,
        "north_count": north,
        "stuck_ge_4_events": stuck_events,
        "stuck_ge_4_total_steps": stuck_ge_4,
        "jump_triggered": jump_triggered,
        "last_10_actions": [(a["step"], a["action"], a["pos"], a["safety_gap"]) for a in last_10],
    }


def main():
    print("=" * 80)
    print("MAZE CRAWLER DIAGNOSTIC: Why does v2 lose 40% against random?")
    print("=" * 80)
    print(f"Games: 20 | Seeds: {SEEDS}")
    print()

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"Running game {i+1}/20 (seed={seed})...", end="", flush=True)
        game = replay_with_action_logging(seed)
        results.append(game)
        status = "ALIVE" if game["factory_alive_until"] == game["total_steps"] else f"DEAD@{game['factory_alive_until']}"
        print(f" {game['result']:4s} | {status} | steps={game['total_steps']}")

    # Summary
    wins = sum(1 for r in results if r["result"] == "WIN")
    losses = sum(1 for r in results if r["result"] == "LOSS")
    draws = sum(1 for r in results if r["result"] == "DRAW")
    print(f"\n{'='*80}")
    print(f"OVERALL: {wins}W - {losses}L - {draws}D ({wins/20*100:.0f}% win rate)")
    print(f"{'='*80}")

    # Detailed logging for each game (every 5 steps)
    print(f"\n{'='*80}")
    print("FACTORY LOG EVERY 5 STEPS")
    print(f"{'='*80}")
    for game in results:
        tag = " *** LOSS ***" if game["result"] == "LOSS" else ""
        print(f"\n--- Game seed={game['seed']} | {game['result']}{tag} "
              f"| Factory alive until step {game['factory_alive_until']}/{game['total_steps']} ---")
        print(f"  {'Step':>4} | {'Pos':>7} | {'Gap':>3} | {'Action':>12} | {'E':>5} | {'MCD':>3} | {'JCD':>3} | {'Moved?':>6}")
        print(f"  {'-'*65}")

        prev_pos = None
        for a in game["factory_actions"]:
            if not a["alive"]:
                continue
            step = a["step"]
            if step % 5 != 0 and step != game["factory_alive_until"] - 1:
                continue
            pos = a["pos"]
            moved = ""
            if prev_pos is not None:
                moved = "YES" if pos != prev_pos else "NO"
            print(f"  {step:>4} | {str(pos):>7} | {a['safety_gap']:>3} | {a['action']:>12} | "
                  f"{a['energy']:>5} | {a['move_cd']:>3} | {a['jump_cd']:>3} | {moved:>6}")
            prev_pos = pos

    # LOSS ANALYSIS
    loss_games = [r for r in results if r["result"] == "LOSS"]
    if loss_games:
        print(f"\n{'='*80}")
        print(f"LOSS ANALYSIS ({len(loss_games)} games)")
        print(f"{'='*80}")

        for game in loss_games:
            analysis = analyze_loss(game)
            print(f"\n  --- Seed {game['seed']} | Factory died at step {analysis['alive_until']} ---")
            print(f"  Death safety_gap:    {analysis['death_safety_gap']}")
            print(f"  Killed by scroll:    {analysis['killed_by_scroll']}")
            print(f"  Steps in danger:     {analysis['danger_steps']} (gap <= 2)")
            print(f"  North progress:      {analysis['north_progress']} rows")
            print(f"  Action breakdown:    {analysis['action_counts']}")
            print(f"  EAST+WEST vs NORTH:  {analysis['east_west_count']} vs {analysis['north_count']}")
            print(f"  Stuck (>=4) events:  {analysis['stuck_ge_4_events']} ({analysis['stuck_ge_4_total_steps']} steps)")
            print(f"  JUMP triggered:      {analysis['jump_triggered']}")
            print(f"  Last 10 actions:")
            for step, action, pos, gap in analysis["last_10_actions"]:
                print(f"    step {step:>3}: {action:>12} at {pos} gap={gap}")

    # CROSS-GAME PATTERNS
    print(f"\n{'='*80}")
    print("CROSS-GAME PATTERNS")
    print(f"{'='*80}")

    # Aggregate action stats for losses vs wins
    for label, games in [("LOSSES", loss_games), ("WINS", [r for r in results if r["result"] == "WIN"])]:
        if not games:
            continue
        all_actions = Counter()
        total_ew = 0
        total_n = 0
        total_jumps = 0
        scroll_deaths = 0
        for game in games:
            a = analyze_loss(game)  # works for wins too
            all_actions.update(a["action_counts"])
            total_ew += a["east_west_count"]
            total_n += a["north_count"]
            total_jumps += a["jump_triggered"]
            if a["killed_by_scroll"]:
                scroll_deaths += 1
        print(f"\n  {label} ({len(games)} games):")
        print(f"    Total action counts: {dict(all_actions)}")
        print(f"    EAST+WEST / NORTH ratio: {total_ew}/{total_n} = {total_ew/max(1,total_n):.2f}")
        print(f"    JUMP triggers: {total_jumps}")
        print(f"    Killed by scroll: {scroll_deaths}")

    # KEY DIAGNOSTIC: How often does factory die with move_cd=0 but still IDLE?
    print(f"\n{'='*80}")
    print("KEY DIAGNOSTIC: Factory death circumstances")
    print(f"{'='*80}")
    for game in loss_games:
        alive_until = game["factory_alive_until"]
        # Look at the last few alive steps
        last_alive = [a for a in game["factory_actions"] if a["alive"]]
        if not last_alive:
            continue
        last = last_alive[-1]
        # Count steps where factory had move_cd=0 but still didn't move
        no_move_when_able = 0
        idle_when_able = 0
        for a in last_alive:
            if a["move_cd"] == 0:
                # Factory could have moved but action was not a move
                if a["action"] not in ("NORTH", "EAST", "WEST", "SOUTH", "JUMP_NORTH"):
                    no_move_when_able += 1
                if a["action"] == "IDLE":
                    idle_when_able += 1
        print(f"  Seed {game['seed']}: died step {alive_until}, last pos={last['pos']}, "
              f"last gap={last['safety_gap']}, last action={last['action']}, "
              f"move_cd=0 but no move: {no_move_when_able}x, IDLE when able: {idle_when_able}x")


if __name__ == "__main__":
    main()
