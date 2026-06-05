"""Compare average reward of two agents vs multiple opponents."""
import sys, os, time, copy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make as _make
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_GAMES = 100
SEEDS = [i * 137 + 42 for i in range(NUM_GAMES)]

def get_state_reset():
    return {
        "turn": 0, "walls": {}, "nodes": set(), "last_factory_pos": None,
        "factory_stuck": 0, "mine_invested": None, "mine_wait": False,
        "mine_wait_since": 0, "last_build_turn": -999,
    }

def load_agent(name):
    if name == "random":
        return "random", None, "random"
    if name.endswith(".py"):
        agent_file = name if os.path.isabs(name) else os.path.join(HERE, name)
    else:
        agent_file = os.path.join(HERE, f"agent_{name}.py")
        if not os.path.exists(agent_file):
            agent_file = os.path.join(HERE, f"agent_submit_{name}.py")

    mod_name = os.path.splitext(os.path.basename(agent_file))[0]
    spec = importlib.util.spec_from_file_location(mod_name, agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    state = getattr(mod, 'STATE', None)
    return mod.agent, state, os.path.basename(agent_file)

def run_games(p0_fn, p0_state, p1_fn, p1_state, label="", num_games=None):
    ng = num_games if num_games is not None else NUM_GAMES
    total_reward = 0
    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    for i in range(ng):
        if p0_state is not None:
            p0_state.clear()
            p0_state.update(get_state_reset())
        if p1_state is not None:
            p1_state.clear()
            p1_state.update(get_state_reset())

        env = _make("crawl", configuration={
            "randomSeed": SEEDS[i],
            "scrollStartInterval": 10,
            "scrollEndInterval": 2,
            "scrollRampSteps": 450,
        }, debug=False)
        env.run([p0_fn, p1_fn])
        steps = env.steps
        final = steps[-1]
        r0 = final[0].reward
        r1 = final[1].reward
        total_reward += r0
        total_steps += len(steps)
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1

    avg_reward = total_reward / ng
    avg_steps = total_steps / ng
    print(f"  {label:20s} | {wins}W-{losses}L-{draws}D | avg_reward={avg_reward:>8.1f} | avg_steps={avg_steps:>6.1f}")
    return avg_reward, avg_steps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare agent rewards")
    parser.add_argument("agents", nargs="+", help="Agent files to compare (e.g. v12 v13)")
    parser.add_argument("--opponents", nargs="+", default=["random", "v1", "v2", "v49", "v50", "v51"],
                        help="Opponents to test against")
    parser.add_argument("--games", type=int, default=100, help="Number of games per matchup (default 100)")
    args = parser.parse_args()
    agents_to_test = [a if a.endswith(".py") else f"agent_{a}.py" for a in args.agents]
    opponents = args.opponents

    print(f"{'Opponent':20s} | ", end="")
    for a in agents_to_test:
        print(f"{a:20s} | ", end="")
    print("diff")
    print("-" * 120)

    for opp in opponents:
        p1_fn, p1_state, opp_label = load_agent(opp)
        results = []
        for agent_file in agents_to_test:
            p0_fn, p0_state, agent_label = load_agent(agent_file)
            avg_r, avg_s = run_games(p0_fn, p0_state, p1_fn, p1_state, label=f"{agent_file} vs {opp_label}", num_games=args.games)
            results.append((avg_r, avg_s))
        if len(results) == 2:
            diff_r = results[1][0] - results[0][0]
            diff_s = results[1][1] - results[0][1]
            sign_r = "+" if diff_r >= 0 else ""
            sign_s = "+" if diff_s >= 0 else ""
            print(f"  {'':20s}   diff = {sign_r}{diff_r:.1f} reward | {sign_s}{diff_s:.1f} steps")
        print()
