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

- **`agent_v12.py`** — Based on agent_v10 with dynamic panic_steps thresholds. Key features over v10: (1) BFS goals expanded to r+2~r+4 (three rows); (2) mine decisions use `panic_steps = gap × scroll_interval` with progressive `ps_safe` threshold: `(10+turn//10)` for turn≤100, `(24+turn//20)` for turn>100; (3) skip JUMP when on friendly mine with stored energy; (4) segmented roi_threshold (50/100/200/9999) by panic_steps (100/50/25). Best local eval: 448 total wins across 6 opponents.

- **`agent_v13.py`** — Based on agent_v12 with late-game survival optimizations. Key changes: (1) removed dead code; (2) BFS-aware worker wall removal prioritizes factory's lateral direction; (3) mid-game mine ROI threshold lowered (roi_threshold=50 for panic_steps 50~100); (4) more workers: up to 3 after turn 300, build even when factory stuck (stuck≥3); (5) Tier 3 lateral movement prefers directions with north exits; (6) urgent mine: build miner when adjacent mining node found (no existing miner, energy≥400, panic_steps>ps_safe); (7) worker transfer: when factory north is wall and worker has energy<80, transfer energy from trapped worker to factory. Eval vs v12: +28W across 6 opponents. Transfer (e<80) adds +10W total vs no-transfer (475W vs 465W across 6 opponents).

- **`agent_v14.py`** — Based on agent_v13 with navigation and survival improvements. Key changes over v13: (1) JUMP skip: when landing point (c,r+2) is an unmined mining node and factory north is clear, skip JUMP to let urgent mine logic handle it; (2) JUMP landing check: skip JUMP_NORTH when landing cell has friendly units, unless emergency (gap<=3 or danger_escape); (3) South fallback (Tier 4): when Tier 1-3 navigation fails, allow MOVE SOUTH if jump_cd>3, preventing factory from staying IDLE when stuck; (4) Worker forward move: when worker near factory, if north is passable but (c,r+1) has a blocking north wall, move north first instead of removing lateral walls; (5) more workers: max 4 after turn 400 (was 3 after turn 300). Eval vs v13: +30W total (504W vs 474W across 6 opponents).

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

### agent_v10 (100-game eval, aggressive scroll: 4→1, 400 ramp)

Key features over v3: (1) 3-tier factory navigation (direct NORTH → pessimistic BFS → unconditional lateral with crystal preference); (2) JUMP landing quality check: require north exit or ≥2 non-south exits at landing (relaxed in dead ends); (3) dead-end detection + panic JUMP: skip Tier 1/2 when r+1 is dead end; when in dead end, allow JUMP to any landing with exits; lateral JUMP escape from dead ends; (4) worker delay: build worker at turn ≥ 4 (after initial JUMP) so worker spawns near factory; stuck build: allow building 2nd worker when stuck ≥ 3; (5) worker E/W wall breaking: when factory stuck, worker breaks E/W walls at dead-end cell; (6) smart crush: crush friendly units unless fresh worker and safe.

| Opponent | Result |
|----------|--------|
| random | 87W-9L-4D (87%) |
| v2 (BFS factory-only) | 87W-10L-3D (87%) |
| v6 (BC+PPO vs fixed) | 78W-20L-2D (78%) |
| v15 (factory-only NN) | 60W-37L-3D (60%) |
| v49 (all-units NN) | 32W-67L-1D (32%) |
| v50 (all-units NN) | 24W-70L-6D (24%) |

### agent_v12 (100-game eval, Kaggle scroll: 10→2, 450 ramp)

Based on agent_v10 (commit 1ed81e0) with BFS goal expansion (r+2~r+4) and dynamic panic_steps thresholds replacing fixed gap thresholds. Key changes over v10: (1) BFS goals expanded from r+2 single row to r+2~r+4 three rows for longer-range pathfinding; (2) all mine-related decisions use `panic_steps = gap × scroll_interval` instead of fixed `gap` thresholds; (3) `ps_safe` (safe threshold) varies by turn: `(10 + turn//10)` for turn≤100, `(24 + turn//20)` for turn>100 — progressively more conservative; (4) `roi_threshold` segmented by panic_steps: 50/100/200/9999 at thresholds 100/50/25; (5) skip JUMP when on friendly mine with stored energy ≥50 and panic_steps > ps_safe.

| Opponent | Result |
|----------|--------|
| random | 100W-0L-0D (100%) |
| v1 (factory-only) | 58W-37L-5D (58%) |
| v2 (BFS factory-only) | 64W-28L-8D (64%) |
| v49 (all-units NN) | 87W-12L-1D (87%) |
| v50 (all-units NN) | 91W-8L-1D (91%) |
| v51 (all-units NN) | 48W-41L-11D (48%) |

vs v11 (previous best): +27W vs v1, +16W vs v2, +5W vs v51, +2W vs v49. avg_reward全面提升（vs v49 +937, vs v50 +895, vs v2 +765）。

### agent_v13 (100-game eval, Kaggle scroll: 10→2, 450 ramp)

Based on agent_v12 with late-game survival optimizations. Key changes over v12: (1) removed dead code; (2) BFS-aware worker wall removal prioritizes factory's lateral direction; (3) mid-game mine ROI threshold lowered (roi_threshold=50 for panic_steps 50~100, was 100); (4) more workers: max 3 after turn 300 (was 2 after turn 400), build even when factory stuck (stuck≥3, energy≥300); (5) Tier 3 lateral movement prefers directions with north exits over crystal preference; (6) urgent mine: build miner when adjacent mining node found (no existing miner, energy≥400, panic_steps>ps_safe); (7) worker transfer: when factory north is wall and worker has energy<80, transfer energy from trapped worker to factory.

| Opponent | eec93ff (no transfer/urgent) | 7a00207 (+urgent mine) | current (+urgent +transfer e<80) |
|----------|------------------------------|------------------------|----------------------------------|
| v1 | 58W-37L-5D, r=621 | 60W-37L-3D, r=851 | 61W-34L-5D, r=927 |
| v2 | 71W-28L-1D, r=727 | 69W-29L-2D, r=930 | 71W-27L-2D, r=974 |
| v15 | — | — | 95W-4L-1D, r=1993 |
| v49 | 94W-5L-1D, r=1785 | 95W-4L-1D, r=2015 | 95W-4L-1D, r=2007 |
| v50 | 94W-6L-0D, r=1723 | 94W-6L-0D, r=1993 | 95W-5L-0D, r=2034 |
| v51 | 59W-35L-6D, r=653 | 53W-41L-6D, r=719 | 57W-37L-6D, r=858 |

Transfer threshold sweep (total wins across 6 opponents): no-transfer 465W, e<60 473W, e<70 472W, **e<80 475W**, e<90 466W.

### agent_v14 (100-game eval, Kaggle scroll: 10→2, 450 ramp)

Based on agent_v13 with navigation and survival improvements. Key changes over v13: (1) JUMP skip on mining node; (2) JUMP_NORTH avoids friendly units on landing (emergency bypass); (3) South fallback (Tier 4, jump_cd>3); (4) Worker moves north first when forward cell has blocking wall; (5) max 4 workers after turn 400.

| Opponent | v13 | v14 | diff |
|----------|-----|-----|------|
| v1 | 61W-34L-5D, r=927 | 71W-25L-4D, r=1349 | +10W, r+423 |
| v2 | 71W-27L-2D, r=974 | 82W-16L-2D, r=1753 | +11W, r+779 |
| v15 | 95W-4L-1D, r=1995 | 93W-5L-2D, r=2395 | -2W, r+400 |
| v49 | 95W-4L-1D, r=2009 | 95W-3L-2D, r=2469 | 0W, r+459 |
| v50 | 95W-5L-0D, r=2034 | 95W-5L-0D, r=2414 | 0W, r+380 |
| v51 | 57W-37L-6D, r=858 | 68W-24L-8D, r=1363 | +11W, r+504 |
| **Total** | **474W** | **504W** | **+30W** |

South fallback jump_cd sweep: >2 (-12W), **>3 (+11W)**, >4 (+11W), >5 (+12W). Selected >3.
