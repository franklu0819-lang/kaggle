"""Compare current vs git-baseline agent_v10 against multiple opponents.
Usage: python compare_v10.py
"""
import sys, os, time, shutil, subprocess, importlib
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make as _make
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
OPPONENTS = ["v2", "v6", "v15", "v49", "v50"]
NUM_GAMES = 100
SEEDS = [i * 137 + 42 for i in range(NUM_GAMES)]

TYPE_FACTORY = 0


def get_state_reset():
    return {
        "turn": 0, "walls": {}, "nodes": set(),
        "last_factory_pos": None, "factory_stuck": 0,
        "mine_invested": None, "mine_wait": False,
        "mine_wait_since": 0, "last_build_turn": -999,
    }


def load_agent(name):
    if name == "random":
        return "random", None
    agent_file = os.path.join(HERE, f"agent_{name}.py")
    if not os.path.exists(agent_file):
        agent_file = os.path.join(HERE, f"agent_submit_{name}.py")
    mod_name = f"agent_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent, getattr(mod, 'STATE', None)


def run_games(agent_fn, agent_state, opp_name):
    opp_fn, opp_state = load_agent(opp_name)
    total_reward = 0
    results = []
    for i in range(NUM_GAMES):
        if agent_state is not None:
            agent_state.clear()
            agent_state.update(get_state_reset())
        if opp_state is not None:
            opp_state.clear()
            opp_state.update(get_state_reset())

        env = _make("crawl", configuration={"randomSeed": SEEDS[i]}, debug=False)
        env.run([agent_fn, opp_fn])
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
        total_reward += r0
        results.append(result)

    w = results.count("WIN")
    l = results.count("LOSS")
    d = results.count("DRAW")
    return w, l, d, total_reward / NUM_GAMES


def main():
    agent_file = os.path.join(HERE, "agent_v10.py")
    backup = agent_file + ".tmp_bak"
    shutil.copy2(agent_file, backup)

    # Phase 1: test CURRENT version
    print("=" * 60)
    print("PHASE 1: CURRENT agent_v10")
    print("=" * 60)
    import agent_v10
    importlib.reload(agent_v10)

    current = {}
    for opp in OPPONENTS:
        w, l, d, avg_r = run_games(agent_v10.agent, agent_v10.STATE, opp)
        current[opp] = (w, l, d, avg_r)
        print(f"  vs {opp}: {w}W-{l}L-{d}D avg_reward={avg_r:.2f}")

    # Phase 2: restore git baseline
    print("\n" + "=" * 60)
    print("PHASE 2: Restoring GIT BASELINE agent_v10")
    print("=" * 60)
    subprocess.run(["git", "checkout", "agent_v10.py"], cwd=HERE, check=True)
    importlib.reload(agent_v10)

    baseline = {}
    for opp in OPPONENTS:
        w, l, d, avg_r = run_games(agent_v10.agent, agent_v10.STATE, opp)
        baseline[opp] = (w, l, d, avg_r)
        print(f"  vs {opp}: {w}W-{l}L-{d}D avg_reward={avg_r:.2f}")

    # Restore modified version
    shutil.copy2(backup, agent_file)
    os.remove(backup)
    print(f"\nRestored current agent_v10.py")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: CURRENT vs BASELINE")
    print("=" * 60)
    for opp in OPPONENTS:
        cw, cl, cd, cr = current[opp]
        bw, bl, bd, br = baseline[opp]
        print(f"\n  vs {opp}:")
        print(f"    CURRENT:  {cw}W-{cl}L-{cd}D  avg_reward={cr:.2f}")
        print(f"    BASELINE: {bw}W-{bl}L-{bd}D  avg_reward={br:.2f}")
        print(f"    DELTA:    {cw-bw:+d}W {cl-bl:+d}L {cd-bd:+d}D  reward={cr-br:+.2f}")

    # Totals
    tc = [sum(x[i] for x in current.values()) for i in range(3)]
    tb = [sum(x[i] for x in baseline.values()) for i in range(3)]
    tcr = sum(x[3] for x in current.values()) / len(OPPONENTS)
    tbr = sum(x[3] for x in baseline.values()) / len(OPPONENTS)
    print(f"\n  TOTAL (across {len(OPPONENTS)} opponents):")
    print(f"    CURRENT:  {tc[0]}W-{tc[1]}L-{tc[2]}D  avg_reward={tcr:.2f}")
    print(f"    BASELINE: {tb[0]}W-{tb[1]}L-{tb[2]}D  avg_reward={tbr:.2f}")
    print(f"    DELTA:    {tc[0]-tb[0]:+d}W {tc[1]-tb[1]:+d}L {tc[2]-tb[2]:+d}D  reward={tcr-tbr:+.2f}")


if __name__ == "__main__":
    main()
