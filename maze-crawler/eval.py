"""Test agent_v1 vs agent_submit_v50, N games with detailed loss analysis."""
import sys, os, time, json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make
import agent_v1
import importlib.util

# Load agent_submit_v50 as separate module
OPPONENT = sys.argv[1] if len(sys.argv) > 1 else "v50"

# Load opponent module
if OPPONENT.startswith("v"):
    opp_file = f"agent_submit_{OPPONENT}.py"
    spec = importlib.util.spec_from_file_location(f"agent_{OPPONENT}", os.path.join(os.path.dirname(os.path.abspath(__file__)), opp_file))
    opp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opp_mod)
    OPPONENT_FN = opp_mod.agent
    OPPONENT_STATE = getattr(opp_mod, 'STATE', None)
else:
    OPPONENT_FN = OPPONENT
    OPPONENT_STATE = None

NUM_GAMES = 100
SEEDS = [i * 137 + 42 for i in range(NUM_GAMES)]


def run_game(seed, game_idx):
    # Reset v1 state
    agent_v1.STATE.update({
        "turn": 0, "walls": {}, "nodes": set(),
        "last_factory_pos": None, "factory_stuck": 0,
        "mine_invested": None, "mine_wait": False, "mine_wait_since": 0,
    })
    # Reset opponent state (if it has one)
    if OPPONENT_STATE is not None:
        OPPONENT_STATE.update({
            "turn": 0, "walls": {}, "nodes": set(),
            "last_factory_pos": None, "factory_stuck": 0,
            "mine_invested": None, "mine_wait": False, "mine_wait_since": 0,
        })

    env = _make("crawl", configuration={"randomSeed": seed}, debug=True)
    env.run([agent_v1.agent, OPPONENT_FN])
    steps = env.steps
    total = len(steps)
    final = steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")

    # Track key metrics
    factory_life = total
    min_energy = 999999
    had_worker = False
    had_mine = False
    worker_build_turn = None
    mine_built_turn = None
    factory_energies = []

    for si in range(total):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        my = {k: v for k, v in robots.items() if v[4] == 0}
        factory_data = next((v for v in my.values() if v[0] == 0), None)
        if factory_data is None:
            factory_life = si
            break
        factory_energy = factory_data[3]
        factory_energies.append(factory_energy)
        min_energy = min(min_energy, factory_energy)
        has_worker = any(v[0] == 2 for v in my.values())
        has_mine = any(v[2] == 0 for v in obs.get("mines", {}).values())
        if has_worker:
            had_worker = True
            if worker_build_turn is None:
                worker_build_turn = si
        if has_mine:
            had_mine = True
            if mine_built_turn is None:
                mine_built_turn = si

    return {
        "seed": seed, "result": result, "r0": r0, "r1": r1,
        "steps": total, "factory_life": factory_life,
        "min_energy": min_energy, "had_worker": had_worker,
        "had_mine": had_mine, "worker_turn": worker_build_turn,
        "mine_turn": mine_built_turn,
        "last_energy": factory_energies[-1] if factory_energies else 0,
    }


def main():
    print(f"Testing agent_v1 vs {OPPONENT} ({NUM_GAMES} games)")
    print(f"Started: {datetime.now().isoformat()}")
    start = time.time()

    results = []
    for i in range(NUM_GAMES):
        r = run_game(SEEDS[i], i + 1)
        results.append(r)
        # Progress
        if (i + 1) % 10 == 0 or r["result"] != "WIN":
            elapsed = time.time() - start
            wins = sum(1 for x in results if x["result"] == "WIN")
            losses = sum(1 for x in results if x["result"] == "LOSS")
            draws = sum(1 for x in results if x["result"] == "DRAW")
            print(f"  [{i+1:3d}/{NUM_GAMES}] {r['result']:4s} seed={r['seed']:5d} "
                  f"steps={r['steps']:3d} life={r['factory_life']:3d} "
                  f"minE={r['min_energy']:6.0f} lastE={r['last_energy']:6.0f} "
                  f"worker={r['had_worker']} mine={r['had_mine']} "
                  f"| {wins}W-{losses}L-{draws}D ({elapsed:.1f}s)")

    # Summary
    wins = sum(1 for r in results if r["result"] == "WIN")
    losses = sum(1 for r in results if r["result"] == "LOSS")
    draws = sum(1 for r in results if r["result"] == "DRAW")

    print(f"\n{'='*60}")
    print(f"FINAL: {wins}W - {losses}L - {draws}D ({wins/NUM_GAMES*100:.1f}%)")
    print(f"Time: {time.time()-start:.1f}s")

    # Loss analysis
    if losses > 0:
        print(f"\n--- LOSS ANALYSIS ({losses} losses) ---")
        loss_results = [r for r in results if r["result"] == "LOSS"]
        for r in loss_results:
            print(f"  seed={r['seed']:5d} steps={r['steps']:3d} life={r['factory_life']:3d} "
                  f"minE={r['min_energy']:6.0f} lastE={r['last_energy']:6.0f} "
                  f"worker={r['had_worker']} workerT={r['worker_turn']} "
                  f"mine={r['had_mine']} mineT={r['mine_turn']}")

        # Patterns
        all_no_mine = sum(1 for r in loss_results if not r["had_mine"])
        all_low_energy = sum(1 for r in loss_results if r["min_energy"] < 100)
        print(f"\n  No mine built: {all_no_mine}/{losses}")
        print(f"  Min energy < 100: {all_low_energy}/{losses}")
        avg_life = sum(r['factory_life'] for r in loss_results) / len(loss_results)
        print(f"  Avg factory life: {avg_life:.1f}")

    # Draw analysis
    if draws > 0:
        print(f"\n--- DRAW ANALYSIS ({draws} draws) ---")
        for r in results:
            if r["result"] == "DRAW":
                print(f"  seed={r['seed']:5d} steps={r['steps']:3d} life={r['factory_life']:3d} "
                      f"minE={r['min_energy']:6.0f} lastE={r['last_energy']:6.0f} "
                      f"worker={r['had_worker']} mine={r['had_mine']}")


if __name__ == "__main__":
    main()
