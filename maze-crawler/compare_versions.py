"""Compare two agent versions on the same seed set."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make

SEEDS = [42, 123, 456, 789, 1001, 2024, 303, 777, 2048, 555,
         111, 222, 333, 444, 5556, 666, 999, 1337, 2025, 3141]


def test_agent(mod, seeds):
    agent_fn = mod.agent
    STATE = mod.STATE
    TYPE_FACTORY = mod.TYPE_FACTORY
    results = []
    for seed in seeds:
        STATE.update({"turn": 0, "walls": {}, "nodes": set(), "mines": {},
                       "enemy_factory": None, "my_factory": None, "enemy_seen": {},
                       "factory_stuck": 0, "factory_last_pos": None})
        fresh = _make("crawl", configuration={"randomSeed": seed}, debug=True)
        fresh.run([agent_fn, "random"])
        steps = fresh.steps
        final = steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        fact_life = len(steps)
        for si in range(len(steps)):
            obs = steps[si][0].observation
            robots = obs.get("robots", {})
            my = {k: v for k, v in robots.items() if v[4] == 0}
            if not any(v[0] == TYPE_FACTORY for v in my.values()):
                fact_life = si
                break
        result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
        results.append((seed, result, r0, fact_life))
    return results


def main():
    import agent_v1
    import agent_v2

    print("Testing v1...")
    v1_results = test_agent(agent_v1, SEEDS)
    print("Testing v2...")
    v2_results = test_agent(agent_v2, SEEDS)

    header = f"{'Seed':>6} | {'v1':>8} | {'v2':>8} | {'Delta':>7} | v1 Res | v2 Res | v1Life | v2Life"
    print(f"\n{header}")
    print("-" * len(header))

    for v1r, v2r in zip(v1_results, v2_results):
        delta = v2r[2] - v1r[2]
        print(f"{v1r[0]:>6} | {v1r[2]:>8.0f} | {v2r[2]:>8.0f} | {delta:>+7.0f} | "
              f"{v1r[1]:>6} | {v2r[1]:>6} | {v1r[3]:>6} | {v2r[3]:>6}")

    for ver, results in [("v1", v1_results), ("v2", v2_results)]:
        wins = sum(1 for r in results if r[1] == "WIN")
        losses = sum(1 for r in results if r[1] == "LOSS")
        draws = sum(1 for r in results if r[1] == "DRAW")
        avg = sum(r[2] for r in results) / len(results)
        life = sum(r[3] for r in results) / len(results)
        print(f"{ver}: {wins}W-{losses}L-{draws}D | Avg reward: {avg:.1f} | Avg life: {life:.1f}")


if __name__ == "__main__":
    main()
