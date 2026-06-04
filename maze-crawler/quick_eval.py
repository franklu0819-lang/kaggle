"""Quick eval: v10 vs key opponents (20 games each), returns W-L-D string."""
import sys, os, time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make as _make
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
N = 30
SEEDS = [i * 137 + 42 for i in range(N)]

def get_state_reset():
    return {
        'turn': 0, 'walls': {}, 'nodes': set(), 'last_factory_pos': None,
        'factory_stuck': 0, 'mine_invested': None, 'mine_wait': False,
        'mine_wait_since': 0, 'last_build_turn': -999,
        'factory_pos_history': deque(maxlen=8),
    }

def load_agent(name):
    if name == 'random': return 'random', None, 'random'
    f = os.path.join(HERE, f"agent_{name}.py")
    if os.path.exists(f): mn = f"agent_{name}"
    else: f = os.path.join(HERE, f"agent_submit_{name}.py"); mn = f"agent_submit_{name}"
    spec = importlib.util.spec_from_file_location(mn, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent, getattr(mod, 'STATE', None), name

def run_game(seed, p0_fn, p0_state, p1_fn, p1_state):
    if p0_state: p0_state.clear(); p0_state.update(get_state_reset())
    if p1_state: p1_state.clear(); p1_state.update(get_state_reset())
    env = _make('crawl', configuration={
        'randomSeed': seed, 'scrollStartInterval': 4,
        'scrollEndInterval': 1, 'scrollRampSteps': 400,
    }, debug=True)
    env.run([p0_fn, p1_fn])
    r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
    return "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")

opp = sys.argv[1] if len(sys.argv) > 1 else 'v49'
p0_fn, p0_state, _ = load_agent('v10')
p1_fn, p1_state, _ = load_agent(opp)

w = l = d = 0
for i in range(N):
    r = run_game(SEEDS[i], p0_fn, p0_state, p1_fn, p1_state)
    if r == "WIN": w += 1
    elif r == "LOSS": l += 1
    else: d += 1

print(f"v10 vs {opp}: {w}W-{l}L-{d}D ({w/N*100:.0f}%)")
