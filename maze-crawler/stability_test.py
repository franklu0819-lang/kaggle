"""Run 50-game stability test for the agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make
from agent_v1 import agent, STATE, TYPE_FACTORY

NUM_GAMES = 500
SEEDS = [i * 137 + 42 for i in range(NUM_GAMES)]


def main():
    results = []
    for i, seed in enumerate(SEEDS):
        STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None, "factory_stuck": 0})
        fresh = _make("crawl", configuration={"randomSeed": seed}, debug=True)
        fresh.run([agent, "random"])
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
        results.append((seed, result, r0, r1, len(steps), fact_life))
        print(f"{i+1:>3}/{NUM_GAMES} | seed={seed:>5} | {result:4s} | "
              f"Our={r0:>7.0f} vs Opp={r1:>7.0f} | Steps={len(steps):>3} | FactLife={fact_life:>3}")

    wins = sum(1 for r in results if r[1] == "WIN")
    losses = sum(1 for r in results if r[1] == "LOSS")
    draws = sum(1 for r in results if r[1] == "DRAW")
    avg_r = sum(r[2] for r in results) / len(results)
    avg_opp = sum(r[3] for r in results) / len(results)
    avg_life = sum(r[5] for r in results) / len(results)
    win_life = [r[5] for r in results if r[1] == "WIN"]
    loss_life = [r[5] for r in results if r[1] == "LOSS"]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"RESULTS: {wins}W - {losses}L - {draws}D ({wins/NUM_GAMES*100:.0f}% win rate)")
    print(f"Avg our reward:    {avg_r:>8.1f}")
    print(f"Avg opp reward:    {avg_opp:>8.1f}")
    print(f"Avg factory life:  {avg_life:>8.1f} steps")
    if win_life:
        print(f"Avg WIN life:      {sum(win_life)/len(win_life):>8.1f} steps ({len(win_life)} games)")
    if loss_life:
        print(f"Avg LOSS life:     {sum(loss_life)/len(loss_life):>8.1f} steps ({len(loss_life)} games)")

    win_rewards = [r[2] for r in results if r[1] == "WIN"]
    loss_rewards = [r[2] for r in results if r[1] == "LOSS"]
    if win_rewards:
        print(f"WIN reward range:  {min(win_rewards):>8.0f} ~ {max(win_rewards):>8.0f}")
    if loss_rewards:
        print(f"LOSS reward range: {min(loss_rewards):>8.0f} ~ {max(loss_rewards):>8.0f}")

    print(f"\nRunning win rate (window=10):")
    for start in range(0, NUM_GAMES, 10):
        window = results[start:start + 10]
        w = sum(1 for r in window if r[1] == "WIN")
        l = sum(1 for r in window if r[1] == "LOSS")
        d = sum(1 for r in window if r[1] == "DRAW")
        avg = sum(r[2] for r in window) / len(window)
        print(f"  Games {start+1:>2}-{start+10:>2}: {w}W-{l}L-{d}D | avg reward: {avg:>7.1f}")


if __name__ == "__main__":
    main()
