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

def run_games(p0_fn, p0_state, p1_fn, p1_state, label=""):
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    for i in range(NUM_GAMES):
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
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1

    avg_reward = total_reward / NUM_GAMES
    print(f"  {label:20s} | {wins}W-{losses}L-{draws}D | avg_reward={avg_reward:>8.1f}")
    return avg_reward

if __name__ == "__main__":
    agents_to_test = ["agent_v11.py", "agent_v12.py"]
    opponents = ["random", "v1", "v2", "v49", "v50", "v51"]

    print(f"{'Opponent':20s} | ", end="")
    for a in agents_to_test:
        print(f"{a:20s} | ", end="")
    print("diff")
    print("-" * 90)

    for opp in opponents:
        p1_fn, p1_state, opp_label = load_agent(opp)
        results = []
        for agent_file in agents_to_test:
            p0_fn, p0_state, agent_label = load_agent(agent_file)
            avg_r = run_games(p0_fn, p0_state, p1_fn, p1_state, label=f"{agent_file} vs {opp_label}")
            results.append(avg_r)
        diff = results[1] - results[0]
        sign = "+" if diff >= 0 else ""
        print(f"  {'':20s}   diff = {sign}{diff:.1f}")
        print()
