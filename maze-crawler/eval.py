"""Eval two agents head-to-head, N games with detailed loss analysis.

Usage:
    python eval.py <opponent>              # agent_v1 vs <opponent>
    python eval.py <agent> <opponent>      # <agent> vs <opponent>

Agent names:
    - Built-in: v1, v2, v3  (loads agent_v{N}.py)
    - Submission: v15, v49, v50, v51  (loads agent_submit_v{N}.py)
    - "random" for random opponent
"""
import sys, os, time, copy
from datetime import datetime
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_GAMES = 100
SEEDS = [i * 137 + 42 for i in range(NUM_GAMES)]

def get_state_reset():
    """Return a fresh copy of state reset values (creates new mutable objects each call)."""
    return {
        "turn": 0,
        "walls": {},
        "nodes": set(),
        "last_factory_pos": None,
        "factory_stuck": 0,
        "mine_invested": None,
        "mine_wait": False,
        "mine_wait_since": 0,
        "last_build_turn": -999,
        "factory_pos_history": deque(maxlen=8),
    }


def load_agent(name):
    """Load an agent module by name. Returns (agent_fn, state_dict_or_None, display_name)."""
    if name == "random":
        return "random", None, "random"

    # If name ends with .py, treat as direct file path
    if name.endswith(".py"):
        agent_file = name if os.path.isabs(name) else os.path.join(HERE, name)
        if not os.path.exists(agent_file):
            print(f"ERROR: file not found: {agent_file}")
            sys.exit(1)
        mod_name = os.path.splitext(os.path.basename(agent_file))[0]
        display = os.path.basename(agent_file)
    else:
        # Try agent_v{N}.py first (v1, v2, v3, ...)
        agent_file = os.path.join(HERE, f"agent_{name}.py")
        if os.path.exists(agent_file):
            mod_name = f"agent_{name}"
            display = name
        else:
            # Try agent_submit_v{N}.py
            agent_file = os.path.join(HERE, f"agent_submit_{name}.py")
            mod_name = f"agent_submit_{name}"
            display = name

        if not os.path.exists(agent_file):
            print(f"ERROR: cannot find agent for '{name}' (tried agent_{name}.py and agent_submit_{name}.py)")
            sys.exit(1)

    spec = importlib.util.spec_from_file_location(mod_name, agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    state = getattr(mod, 'STATE', None)
    return mod.agent, state, display


def run_game(seed, p0_fn, p0_state, p1_fn, p1_state):
    # Reset states with fresh mutable objects (prevents cross-game data leakage)
    if p0_state is not None:
        p0_state.clear()
        p0_state.update(get_state_reset())
    if p1_state is not None:
        p1_state.clear()
        p1_state.update(get_state_reset())

    env = _make("crawl", configuration={
        "randomSeed": seed,
        "scrollStartInterval": 10,
        "scrollEndInterval": 2,
        "scrollRampSteps": 450,
    }, debug=True)
    env.run([p0_fn, p1_fn])
    steps = env.steps
    total = len(steps)
    final = steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")

    # Track key metrics for player 0
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
    # Parse args
    if len(sys.argv) == 1:
        print("Usage: python eval.py <agent> [opponent]")
        print("  python eval.py v50              # v1 vs v50")
        print("  python eval.py v3 v50           # v3 vs v50")
        print("  python eval.py v1 random        # v1 vs random")
        print("  python eval.py agent_submit_v1.py random   # direct file path")
        sys.exit(1)

    opp_name = sys.argv[-1]
    agent_name = sys.argv[1] if len(sys.argv) > 2 else "v1"

    p0_fn, p0_state, agent_label = load_agent(agent_name)
    p1_fn, p1_state, opp_label = load_agent(opp_name)

    print(f"Testing {agent_label} vs {opp_label} ({NUM_GAMES} games)")
    print(f"Started: {datetime.now().isoformat()}")
    start = time.time()

    results = []
    for i in range(NUM_GAMES):
        r = run_game(SEEDS[i], p0_fn, p0_state, p1_fn, p1_state)
        results.append(r)
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
    print(f"{agent_label} vs {opp_label}: {wins}W - {losses}L - {draws}D ({wins/NUM_GAMES*100:.1f}%)")
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
