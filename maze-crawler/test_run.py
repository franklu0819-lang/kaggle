"""
Maze Crawler - Multi-game test runner with detailed logging.
Runs N games and aggregates results for analysis.
"""

import json
import sys
import os
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_v1 import (
    agent as fog_agent, STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER
)

TYPE_NAMES = {TYPE_FACTORY: "Factory", TYPE_SCOUT: "Scout",
              TYPE_WORKER: "Worker", TYPE_MINER: "Miner"}


def run_single_game(env, agent_fn, opponent, seed, game_idx, log_lines):
    """Run a single game and return detailed stats."""

    # Reset agent state between games
    STATE["turn"] = 0
    STATE["walls"] = {}
    STATE["nodes"] = set()
    STATE["mines"] = {}
    STATE["enemy_factory"] = None
    STATE["my_factory"] = None
    STATE["enemy_seen"] = {}
    STATE["factory_stuck"] = 0
    STATE["factory_last_pos"] = None

    log = log_lines.append
    log(f"\n{'='*80}")
    log(f"GAME {game_idx} — seed={seed}")
    log(f"{'='*80}")

    start = time.time()
    # Create a fresh env for each game to ensure seed works
    from kaggle_environments import make as _make
    fresh_env = _make("crawl", configuration={"randomSeed": seed}, debug=True)

    fresh_env.run([agent_fn, opponent])
    env = fresh_env
    elapsed = time.time() - start

    steps = env.steps
    total_steps = len(steps)
    final = steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    s0, s1 = final[0].status, final[1].status

    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
    log(f"Result: {result} | Our={r0} vs Opp={r1} | Steps={total_steps} | Time={elapsed:.2f}s")
    log(f"  Our status: {s0} | Opp status: {s1}")

    # Track per-step metrics for our agent (player 0)
    turn_log = []
    factory_alive_until = total_steps
    first_scout_step = None
    first_worker_step = None
    first_miner_step = None
    first_mine_step = None
    max_units = 0
    max_energy = 0
    mines_built = 0

    for si in range(total_steps):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        my = {k: v for k, v in robots.items() if v[4] == 0}
        enemy = {k: v for k, v in robots.items() if v[4] != 0}

        my_energy = sum(v[3] for v in my.values())
        counts = defaultdict(int)
        for v in my.values():
            counts[TYPE_NAMES.get(v[0], f"T{v[0]}")] += 1

        has_factory = any(v[0] == TYPE_FACTORY for v in my.values())
        if not has_factory and factory_alive_until == total_steps:
            factory_alive_until = si

        my_mines = sum(1 for v in obs.get("mines", {}).values() if v[2] == 0)
        nodes_count = len(obs.get("miningNodes", {}))
        south = obs.get("southBound", 0)
        north = obs.get("northBound", 0)

        if counts.get("Scout", 0) > 0 and first_scout_step is None:
            first_scout_step = si
        if counts.get("Worker", 0) > 0 and first_worker_step is None:
            first_worker_step = si
        if counts.get("Miner", 0) > 0 and first_miner_step is None:
            first_miner_step = si
        if my_mines > 0 and first_mine_step is None:
            first_mine_step = si

        max_units = max(max_units, len(my))
        max_energy = max(max_energy, my_energy)
        mines_built = max(mines_built, my_mines)

        turn_log.append({
            "step": si, "energy": my_energy, "units": dict(counts),
            "total_units": len(my), "enemies_visible": len(enemy),
            "mines": my_mines, "nodes": nodes_count,
            "south": south, "north": north,
            "factory_row": next((v[2] for v in my.values() if v[0] == TYPE_FACTORY), None),
            "factory_alive": has_factory,
        })

    # Log sampled turns
    sample_interval = max(1, total_steps // 12)
    log(f"\n  Step-by-step (every {sample_interval} steps):")
    log(f"  {'Step':>5} | {'Energy':>6} | {'Units':>20} | {'Factory':>8} | "
        f"{'Mines':>5} | {'Nodes':>5} | {'South':>5} | {'North':>5}")
    log(f"  {'-'*80}")

    for t in turn_log:
        if t["step"] % sample_interval == 0 or t["step"] == total_steps - 1:
            units_str = ", ".join(f"{k}:{v}" for k, v in sorted(t["units"].items()))
            fact_str = f"row {t['factory_row']}" if t["factory_row"] is not None else "DEAD"
            log(f"  {t['step']:>5} | {t['energy']:>6} | {units_str:>20} | {fact_str:>8} | "
                f"{t['mines']:>5} | {t['nodes']:>5} | {t['south']:>5} | {t['north']:>5}")

    # Log key milestones
    log(f"\n  Milestones:")
    log(f"    Factory survived until step: {factory_alive_until}")
    log(f"    First scout built: step {first_scout_step}" if first_scout_step else "    No scout built")
    log(f"    First worker built: step {first_worker_step}" if first_worker_step else "    No worker built")
    log(f"    First miner built: step {first_miner_step}" if first_miner_step else "    No miner built")
    log(f"    First mine built: step {first_mine_step}" if first_mine_step else "    No mine built")
    log(f"    Max units: {max_units}")
    log(f"    Max energy: {max_energy}")
    log(f"    Total mines: {mines_built}")

    return {
        "result": result, "our_reward": r0, "opp_reward": r1,
        "steps": total_steps, "time": elapsed,
        "factory_alive_until": factory_alive_until,
        "first_scout": first_scout_step, "first_worker": first_worker_step,
        "first_miner": first_miner_step, "first_mine": first_mine_step,
        "max_units": max_units, "max_energy": max_energy,
        "mines_built": mines_built, "turn_log": turn_log,
    }


def main():
    from kaggle_environments import make

    NUM_GAMES = 10
    SEEDS = [42, 123, 456, 789, 1001, 2024, 303, 777, 2048, 555]

    log_lines = []
    log = log_lines.append

    log(f"Maze Crawler — Multi-Game Test Report")
    log(f"Date: {datetime.now().isoformat()}")
    log(f"Games: {NUM_GAMES} | Opponent: random")
    log(f"Agent: Fog-Aware Multi-Agent Strategy")
    log(f"=" * 80)

    env = make("crawl", configuration={"randomSeed": 42}, debug=True)
    log(f"\nConfiguration:")
    for k, v in sorted(env.configuration.items()):
        log(f"  {k}: {v}")

    results = []
    for i in range(NUM_GAMES):
        seed = SEEDS[i] if i < len(SEEDS) else i * 1000 + 7
        r = run_single_game(env, fog_agent, "random", seed, i + 1, log_lines)
        results.append(r)

    # Aggregate summary
    log(f"\n{'='*80}")
    log(f"AGGREGATE SUMMARY ({NUM_GAMES} games)")
    log(f"{'='*80}")

    wins = sum(1 for r in results if r["result"] == "WIN")
    losses = sum(1 for r in results if r["result"] == "LOSS")
    draws = sum(1 for r in results if r["result"] == "DRAW")

    log(f"  Record: {wins}W - {losses}L - {draws}D")
    log(f"  Avg our reward:   {sum(r['our_reward'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Avg opp reward:   {sum(r['opp_reward'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Avg steps:        {sum(r['steps'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Avg factory life: {sum(r['factory_alive_until'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Avg max units:    {sum(r['max_units'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Avg max energy:   {sum(r['max_energy'] for r in results) / NUM_GAMES:.1f}")
    log(f"  Total mines built: {sum(r['mines_built'] for r in results)}")

    log(f"\n  Per-game results:")
    log(f"  {'Game':>4} | {'Result':>5} | {'Our':>6} | {'Opp':>6} | {'Steps':>5} | "
        f"{'FactLife':>8} | {'MaxU':>4} | {'Mines':>5}")
    log(f"  {'-'*65}")
    for i, r in enumerate(results):
        log(f"  {i+1:>4} | {r['result']:>5} | {r['our_reward']:>6} | {r['opp_reward']:>6} | "
            f"{r['steps']:>5} | {r['factory_alive_until']:>8} | {r['max_units']:>4} | {r['mines_built']:>5}")

    # Diagnosis
    log(f"\n--- Diagnosis ---")
    avg_factory_life = sum(r["factory_alive_until"] for r in results) / NUM_GAMES
    if avg_factory_life < 100:
        log(f"  ISSUE: Factory dies early (avg {avg_factory_life:.0f} steps)")
        log(f"  -> Factory not moving north fast enough to escape scroll")
    if sum(r["mines_built"] for r in results) == 0:
        log(f"  ISSUE: No mines built in any game")
        log(f"  -> Miner never reaches mining node, or never built")
    if all(r["first_scout"] is None for r in results):
        log(f"  ISSUE: No scouts ever built")
    if all(r["first_worker"] is None for r in results):
        log(f"  ISSUE: No workers ever built")

    # Output
    print("\n".join(log_lines))

    # Save to file
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nFull log saved to: {log_path}")

    # Save JSON data for further analysis
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_data.json")
    json_data = [{k: v for k, v in r.items() if k != "turn_log"} for r in results]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Game data saved to: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
