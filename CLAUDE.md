# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle "Maze Crawler" competition agent. A factory unit navigates a procedurally-generated scrolling maze (Eller algorithm, mirrored left/right halves) to survive as long as possible against an opponent. The map scrolls south-to-north at increasing speed, and units left behind the south boundary die.

## Commands

All commands run from `maze-crawler/maze-crawler/`:

```bash
# Evaluate agent_v1 against a specific opponent (100 games)
python eval_v1.py <version>   # e.g., eval_v1.py v15, eval_v1.py v49, eval_v1.py v50

# Run multi-game test (10 games, detailed per-game logging)
python test_run.py

# Run stability test (500 games, aggregate stats only)
python stability_test.py

# Compare two agent versions head-to-head
python compare_versions.py

# Train v3: REINFORCE + Reward Shaping (auto-versions weights)
python -u train.py

# Train v4: BC Pre-train + PPO + Reward Shaping
python -u train_v2.py

# Train v5: BC + PPO + Self-Play (70% self-play + 30% random)
python -u train_v3.py

# Train v6: BC + PPO vs fixed opponents (50% v3 + 30% v5 + 20% random)
python -u train_v6.py

# Export weights + package submission file
python -c "
import torch, numpy as np
sd = torch.load('nn_weights_vN.pt')
key_map = {
    'backbone.0.weight': 'net.0.weight', 'backbone.0.bias': 'net.0.bias',
    'backbone.2.weight': 'net.2.weight', 'backbone.2.bias': 'net.2.bias',
    'policy_head.weight': 'net.4.weight', 'policy_head.bias': 'net.4.bias',
}
with open('nn_weights.py', 'w') as f:
    f.write('import numpy as np\nWEIGHTS = {\n')
    for ok, nk in key_map.items():
        f.write(f\"    '{nk}': np.array({sd[ok].detach().numpy().tolist()}, dtype=np.float32),\n\")
    f.write('}\n')
"
# Then combine: replace 'from nn_weights import WEIGHTS' in agent_v3.py with nn_weights.py content

# Submit to Kaggle (needs proxy)
HTTPS_PROXY=http://127.0.0.1:7890 kaggle competitions submit -c maze-crawler -f agent_submit_vN.py -m "message"
```

All tests use `kaggle_environments` with the "crawl" environment. The agent is always player 0 vs "random" opponent.

## Architecture

### Agent Versions

- **`agent_v1.py`** — Rule-based agent with aggressive JUMP and BFS max_nodes=500. Optimistic pathfinding: unknown cells treated as passable. Units processed in strict priority order (scouts → workers → miners → factory). Includes enemy factory threat avoidance: cooldown-gated danger zones prevent factory-factory collisions (mutual destruction loses tiebreaker). Dynamic mine ROI with panic_steps-based thresholds.

- **`agent_v2.py`** — Fog-aware rule-based agent with full persistent state (wall memory, enemy tracking, mine tracking). Uses conservative pathfinding: unknown cells are treated as walled (pessimistic BFS). Complex factory decision tree with stuck detection, diagonal exploration, and south-backtrack fallback. Non-factory units have attack, transfer, and mine-recharge behaviors. Also used as BC expert data source.

- **`agent_v3.py`** — Optimized rule-based agent (factory-focused). Key features: (1) no mine_wait IDLE or mine collection IDLE; (2) worker threshold E≥1500+gap≥8; (3) simplified JUMP (no landing quality filter); (4) simplified MOVE (v2-style tiers + enemy danger avoidance for collision prevention); (5) BFS goals always northward (no mine target diversion). This is the active development agent.

### Submission Files

- **`eval_v1.py`** — Evaluation script for agent_v1 vs a specific opponent version. Runs 100 games with fixed seeds, reports W/L/D and per-loss seed details. Usage: `python eval_v1.py <version>` where version maps to `agent_submit_v{N}.py`.
- **`agent_submit_v2.py`** — Previous Kaggle submission (v2 baseline)
- **`agent_submit_v3.py`** — Kaggle submission (REINFORCE v3 weights)
- **`agent_submit_v4.py`** — Kaggle submission (BC+PPO v4 weights)
- **`agent_submit_v5.py`** — Kaggle submission (BC+PPO+SelfPlay v5 weights)
- **`agent_submit_v6.py`** — Current Kaggle submission (BC+PPO vs fixed opponents, v6 weights)

Self-contained bundles combining `agent_v3.py` logic + embedded weights (~576KB). Weight keys must be mapped from PyTorch names (`backbone.*`, `policy_head.*`) to numpy inference names (`net.*`).

### Training Scripts

- **`train.py`** — REINFORCE + per-step shaped rewards. Produces versioned weights: `nn_weights_v{N}.pt` (best), `nn_weights_v{N}_final.pt`, `nn_weights_v{N}.py` (exported).
- **`train_v2.py`** — BC pre-training (from agent_v2 expert data) + PPO fine-tuning with GAE advantage estimation, clipped surrogate objective, entropy regularization, and value baseline. Opponent: random.
- **`train_v3.py`** — Same as train_v2 but with self-play: 70% of games use same-model greedy opponent, 30% random. Prevents strategy collapse via mixing.
- **`train_v6.py`** — Same as train_v2 but with fixed opponents: 50% v3 + 30% v5 + 20% random. Loads agent_submit_v3.py and agent_submit_v5.py as separate modules with independent STATE dicts.
- **`train_v10.py`** — Factory-only NN PPO training (9-action space, 137-dim input). Command line args: `UNIT_WEIGHT VERSION NUM_ITER`.
- **`train_v11.py`** — All-unit NN PPO training (13-action unified space, 149-dim input). Shared network controls all unit types with type-specific masking.
- **`train_v15.py`** — All-unit NN PPO training with v10-based reward + per-unit shaping. Args: `VERSION NUM_ITER DELTA_UNITS`. Reward: delta_e/1000 + delta_gap×1.0 + delta_units×W + survival +0.01 + terminal +5/-1. Shaping: REMOVE +0.05, TRANSFORM +0.1, scout_ahead +0.01.
- **`train_v16.py`** — Same as train_v15 but REINFORCE (no value baseline, Monte Carlo returns).

### Key Game Mechanics (from analysis.md)

- **Map**: 20-wide, mirrored halves, fixed center wall between cols 9-10 with occasional doors (8% per row)
- **Scroll speed**: starts every 4 steps, accelerates to every 1 step. Factory max speed is 0.5 cells/step — JUMP (2 cells, 20-turn cooldown) is essential to stay ahead
- **Unit types**: Factory (str 4, ∞ energy), Scout (50 cost, str 1), Worker (200 cost, str 2), Miner (300 cost, str 3)
- **Combat**: higher strength crushes lower; equal strength = mutual kill; only enemy Factory can kill your Factory
- **Miner → TRANSFORM** on mining nodes creates energy-generating mines (50/turn)

### Pathfinding

Two approaches exist:
- **Pessimistic** (`agent_v2.py`): `blocked()` treats unseen cells as walls. BFS only through known passable cells. Fallback: `known_blocked()` allows unknown cells for greedy exploration.
- **Optimistic** (`agent_v1.py`): `can_go()` treats unseen cells as passable. BFS explores aggressively but may hit actual walls.

### Enemy Factory Threat Avoidance (agent_v1.py)

`get_enemy_factory_threat()` computes two zones for each visible enemy factory:
- **Hard block**: Enemy factory current cell. NEVER entered — mutual destruction loses tiebreaker (total energy, then unit count).
- **Danger**: Cells enemy could reach NEXT turn. Only includes MOVE neighbors when `move_cd==0` and JUMP landings when `jump_cd==0`. Cooldown gating is critical — without it, the factory retreats from distant enemies causing oscillation and scroll-out regression.

These zones are used in `factory_try_move()` (hard_block always rejected, danger rejected unless `allow_danger=True`) and `factory_action()` (panic mode when `gap<=3` or `must_escape`). The JUMP section has a `danger_escape` trigger when all MOVE targets are dangerous.

### Reward Shaping

**Factory-only training (train.py through train_v10.py):**

5 per-step reward components + discounted returns (gamma=0.99):
- Gap reward: `(factory_row - southBound) / 20` × W_GAP
- Move reward: `delta_row` × W_MOVE (encourages northward movement)
- Jump reward: W_JUMP × (effective +1.0 / partial +0.3 / wasted -0.5)
- Survival reward: W_SURVIVAL (per-step bonus)
- Outcome reward: terminal only, WIN +3.0 / LOSS -1.0

**All-unit training (train_v11.py through train_v16.py):**

Team reward (shared across all units per step):
- `delta_total_energy / 1000` — energy changes including unit deaths, mine income
- `delta_gap × 1.0` — factory progress vs scroll boundary
- `delta_units × W` — unit count changes (W = 0.1 or 0.2)
- `+ 0.01` survival bonus

Per-unit behavioral shaping:
- Worker REMOVE: +0.05
- Miner TRANSFORM: +0.1
- Scout ahead of factory: +0.01

Terminal: +5 (win) / -1 (loss) / 0 (draw)

## State Management

Agents use module-level `STATE` dicts that persist across turns within a single game. **Must reset STATE between games** in test runners — each test file has its own reset logic. The key fields: `turn`, `walls`, `nodes`, `mines`, `enemy_seen`, `factory_stuck`, `factory_last_pos`.

When using fixed opponents (train_v6.py), each opponent module has its own independent STATE dict that must be reset before each game.

## Reference

- **`reference/README.md`** — Game rules (English)
- **`reference/README_zh.md`** — Game rules (Chinese translation)
- **`reference/AGENTS.md`** — Agent API documentation
- **`reference/main.py`** — Starter agent example

## State Management (old)

## Kaggle Submission Constraints

- No PyTorch at inference time — `agent_v3.py` uses pure numpy for the forward pass
- `nn_weights.py` is a Python file containing serialized numpy arrays (~576KB)
- The agent function signature is `agent(obs, config) -> dict[str, str]` mapping unit UIDs to action strings
- Kaggle auth via `~/.kaggle/access_token` (kagglehub) or `~/.kaggle/kaggle.json` (kaggle CLI)

## Training Results

### Factory-only NN (9-action, 137-dim input)

| Version | Method | Best WR (50-game batch) | 500-game Eval vs Random | vs v3 Head-to-Head |
|---------|--------|------------------------|------------------------|--------------------|
| v3 | REINFORCE + Reward Shaping | 96% | 77.0% (385W-99L-16D) | — |
| v4 | BC + PPO vs random | 90% | 77.4% (387W-99L-14D) | — |
| v5 | BC + PPO + SelfPlay (70/30) | 70% | 81.2% (406W-80L-14D) | 55.6% (278W-207L-15D) |
| v6 | BC + PPO vs 50%v3+30%v5+20%rand | 84% | 83.0% (415W-75L-10D) | 83.0% (415W-75L-10D) |

### All-unit NN (13-action, 149-dim input)

| Version | Method | Reward | Best WR | 500-game Eval |
|---------|--------|--------|---------|---------------|
| v25 | PPO, delta_gap×0.5, terminal +5/-1 | v11 reward | 67% | 64.4% (322W-124L-54D) |
| v14 | PPO, delta_gap×0.1, delta_units×0.05 | v15 weak | 51% | — |
| v15 | REINFORCE, same as v14 | v15 weak | 47% | — |
| v16 | PPO, delta_gap×1.0, delta_units×0.1 | v10-based | training... | — |
| v17 | PPO, delta_gap×1.0, delta_units×0.2 | v10-based | training... | — |

### agent_v1 vs NN Opponents (100-game eval, after enemy threat avoidance fix)

| Opponent | Before Fix | After Fix | Change |
|----------|-----------|-----------|--------|
| v15 (factory-only NN) | 91W-7L-2D | 94W-6L-0D | +3W |
| v49 (all-units NN) | 92W-6L-2D | 93W-5L-2D | +1W |
| v50 (all-units NN) | 96W-4L-0D | 96W-4L-0D | unchanged |
| **Total** | **279W-17L-4D** | **283W-15L-2D** | **+4W net** |

Key: 2 enemy-factory-collision losses fixed (seeds 6344, 7988). 15 remaining losses are all scroll-out (mechanical ceiling: factory speed < late-game scroll speed).

### agent_v1 vs Strong Opponents (100-game eval, current baseline)

| Opponent | Result |
|----------|--------|
| random | 100W-0L-0D |
| v15 (factory-only NN) | 95W-5L-0D |
| v49 (all-units NN) | 91W-8L-1D |
| v50 (all-units NN) | 94W-6L-0D |

agent_v3 changes (mine overhead 3→2, worker threshold 600/800, JUMP landing filter) are neutral: within ±3W of agent_v1 across all opponents. agent_v1 remains the current best.

### agent_v3 Optimized (100-game eval)

Key changes from original v3: (1) worker threshold raised to E≥1500+gap≥8 (was E≥500); (2) mine_wait IDLE removed; (3) mine target removed from BFS goals; (4) JUMP simplified (no landing quality filter, no danger check); (5) MOVE simplified (v2-style tiers with danger avoidance for collision prevention).

| Opponent | Result |
|----------|--------|
| random | 100W-0L-0D |
| v2 (BFS factory-only) | 48W-42L-10D |
| v15 (factory-only NN) | 91W-8L-1D |
| v50 (all-units NN) | 94W-6L-0D |

vs original v3: +15W vs v2 (33→48), -4W vs v15 (95→91), 0 vs v50 (94→94). v3 now outperforms v1 vs v2 (was losing 63%) at cost of -4W vs v15.

### Loss Analysis (19 losses across v15/v49/v50)

All 19 losses are scroll-out deaths (avg life ~437 steps). Key patterns:
- Avg stuck time: 152 steps/game (35% of factory life)
- No mine built: 15/19 losses
- Factory typically reaches gap=15-19 around step 200, then slowly scrolls out

See `loss_analysis.md` for full details.

### agent_v10 (100-game eval, current best)

Key features over v3: (1) 3-tier factory navigation (direct NORTH → pessimistic BFS → unconditional lateral with crystal preference); (2) on_own_mine: stay on own mine when panic_steps>60 (turn<100) or panic_steps>100 (turn 100-200), earning energy from mines when safe; (3) dead-end detection: skip Tier 1/2 when r+1 is complete dead end (no north/east/west exits) and jump_cd>6, forcing lateral movement instead; (4) worker prioritizes wall removal at factory front; (5) smart crush: crush friendly units unless fresh worker and safe.

| Opponent | Result |
|----------|--------|
| v2 (BFS factory-only) | 81W-12L-7D |
| v6 (BC+PPO vs fixed) | 75W-21L-4D |
| v15 (factory-only NN) | 96W-4L-0D |
| v49 (all-units NN) | 96W-2L-2D |
| v50 (all-units NN) | 96W-4L-0D |
