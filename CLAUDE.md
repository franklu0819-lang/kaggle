# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle "Maze Crawler" competition agent. A factory unit navigates a procedurally-generated scrolling maze (Eller algorithm, mirrored left/right halves) to survive as long as possible against an opponent. The map scrolls south-to-north at increasing speed, and units left behind the south boundary die.

## Commands

All commands run from `maze-crawler/maze-crawler/`:

```bash
# Run multi-game test (10 games, detailed per-game logging)
python test_run.py

# Run stability test (500 games, aggregate stats only)
python stability_test.py

# Compare two agent versions head-to-head
python compare_versions.py

# Train v3: REINFORCE + Reward Shaping (auto-versions weights)
python -u train.py

# Train v4+: BC Pre-train + PPO + Reward Shaping
python -u train_v2.py

# Evaluate a trained model (500 games)
python -c "from train import *; import torch; net=PolicyNet(); net.load_state_dict(torch.load('nn_weights_v3.pt')); evaluate(net)"

# Export weights + package submission file
python -c "from train_v2 import *; import torch; m=ActorCritic(); m.load_state_dict(torch.load('nn_weights_v4.pt')); export_weights(m, 4, path='nn_weights.py')"
# Then combine agent_v3.py + nn_weights.py into agent_submit_v4.py

# Submit to Kaggle (needs proxy)
HTTPS_PROXY=http://127.0.0.1:7890 kaggle competitions submit -c maze-crawler -f agent_submit_v4.py -m "message"
```

All tests use `kaggle_environments` with the "crawl" environment. The agent is always player 0 vs "random" opponent.

## Architecture

### Agent Versions

- **`agent_v1.py`** — Simplified rule-based agent. Optimistic pathfinding: unknown cells treated as passable. No persistent wall/enemy memory between turns. Units processed in strict priority order (scouts → workers → miners → factory). Used as the NN training base.

- **`agent_v2.py`** — Fog-aware rule-based agent with full persistent state (wall memory, enemy tracking, mine tracking). Uses conservative pathfinding: unknown cells are treated as walled (pessimistic BFS). Complex factory decision tree with stuck detection, diagonal exploration, and south-backtrack fallback. Non-factory units have attack, transfer, and mine-recharge behaviors. Also used as BC expert data source.

- **`agent_v3.py`** — NN hybrid agent: factory decisions via a trained neural network (3-layer MLP, softmax over 9 actions), all other units use rule-based logic from `agent_v1.py`. Weights loaded from `nn_weights.py`. This is the submission agent template.

### Submission Files

- **`agent_submit_v2.py`** — Previous Kaggle submission (v2 baseline)
- **`agent_submit_v4.py`** — Current Kaggle submission (BC+PPO, v4 weights embedded)

Self-contained bundles combining `agent_v3.py` logic + `nn_weights.py` weights (~576KB).

### Training Scripts

- **`train.py`** — REINFORCE + per-step shaped rewards. Produces versioned weights: `nn_weights_v{N}.pt` (best), `nn_weights_v{N}_final.pt`, `nn_weights_v{N}.py` (exported).
- **`train_v2.py`** — BC pre-training (from agent_v2 expert data) + PPO fine-tuning with GAE advantage estimation, clipped surrogate objective, entropy regularization, and value baseline.

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

### Reward Shaping (shared by both training scripts)

5 per-step reward components + discounted returns (gamma=0.99):
- Gap reward: `(factory_row - southBound) / 20` × W_GAP
- Move reward: `delta_row` × W_MOVE (encourages northward movement)
- Jump reward: W_JUMP × (effective +1.0 / partial +0.3 / wasted -0.5)
- Survival reward: W_SURVIVAL (per-step bonus)
- Outcome reward: terminal only, WIN +3.0 / LOSS -1.0

## State Management

Agents use module-level `STATE` dicts that persist across turns within a single game. **Must reset STATE between games** in test runners — each test file has its own reset logic. The key fields: `turn`, `walls`, `nodes`, `mines`, `enemy_seen`, `factory_stuck`, `factory_last_pos`.

## Kaggle Submission Constraints

- No PyTorch at inference time — `agent_v3.py` uses pure numpy for the forward pass
- `nn_weights.py` is a Python file containing serialized numpy arrays (~576KB)
- The agent function signature is `agent(obs, config) -> dict[str, str]` mapping unit UIDs to action strings
- Kaggle auth via `~/.kaggle/access_token` (kagglehub) or `~/.kaggle/kaggle.json` (kaggle CLI)

## Training Results

| Version | Method | Best WR (50-game batch) | 500-game Eval |
|---------|--------|------------------------|---------------|
| v3 | REINFORCE + Reward Shaping | 96% | 77.0% (385W-99L-16D) |
| v4 | BC + PPO + Reward Shaping | 90% | 77.4% (387W-99L-14D) |
