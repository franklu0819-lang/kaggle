# CHANGELOG — Maze Crawler Agent 版本记录

> 记录规则 agent 各版本的核心差异与演进思路。

---

## v14 → v15: Tier 0 悲观BFS + 导航参数预计算

**核心变更：**

| 项目 | v14 | v15 |
|------|-----|-----|
| 导航参数计算时机 | JUMP之后才计算 | JUMP之前预计算（move到mine之间） |
| Tier 0（新增） | 无 | 悲观BFS，窄搜索（3列: c±1），目标 r+3~r+6 |
| Tier 2 BFS策略 | 悲观BFS (`can_go_pessimistic`) | 乐观BFS (`can_go`) |
| 紧急矿场 (urgent_mine) | 无turn限制 | 增加 `turn < 300` 条件 |
| JUMP跳过条件 | 仅检查landing可见 | 双重检查：landing可见 + 路径可达（北2格皆无墙） |

**设计动机：**
- 新增 Tier 0 是为了避免工厂走弯路。悲观BFS只考虑已知可通行的路径，目标设在远处（r+3~r+6），确保工厂始终朝着有明确前路的远方向走。
- 导航参数预计算让 Tier 0 能在 JUMP 之前执行，避免在 JUMP 优先级下错过更好的移动选择。

---

## v15 → v16: bfs_first_step 返回元组 + Tier 0 路径长度过滤

**核心变更：**

| 项目 | v15 | v16 |
|------|-----|-----|
| `bfs_first_step` 返回值 | 方向字符串或 None | **`(first_dir, path_length)` 元组**或 `(None, 0)` |
| Tier 0 路径过滤 | 无路径长度过滤 | `path_len <= 4` 且 `dc2 != 0`（仅横向） |
| Tier 0 目标范围 | r+3~r+6 | r+3~r+4（收紧） |
| 紧急矿场turn限制 | `turn < 300` | 放宽为 `turn < 400` |
| Tier 2 BFS策略 | 乐观BFS (`can_go`) | 乐观BFS (`can_go`)（未变） |

**`bfs_first_step` 签名变更说明：**

v16起，`bfs_first_step` 返回元组 `(first_dir, path_length)` 而非单个方向字符串。所有调用点都已更新为解构：
```python
# v15 及之前
step_dir = bfs_first_step(start, goals, obs, config, passable_fn)

# v16 起
step_dir, path_len = bfs_first_step(start, goals, obs, config, passable_fn)
# 或不需要 path_len 时
step_dir, _ = bfs_first_step(start, goals, obs, config, passable_fn)
```

**设计动机：**
- Tier 0 收紧为仅接受路径长度 ≤4 且方向为横向（`dc2 != 0`）的结果，避免工厂走短视的北移（浪费Tier 0 的高优先级）。
- 目标范围从 r+3~r+6 缩至 r+3~r+4，使搜索更聚焦于近期的可确认通路。

---

## v16 → v17: Tier 0/2 全面悲观BFS + 紧急矿场收紧

**核心变更：**

| 项目 | v16 | v17 |
|------|-----|-----|
| Tier 0 路径过滤 | `path_len <= 4` + `dc2 != 0`（仅横向） | `path_len <= 5` + `dr2 > 0`（仅北向） |
| Tier 0 目标范围 | r+3~r+4 | r+3~r+4（不变） |
| Tier 2 BFS策略 | 乐观BFS (`can_go`) | **悲观BFS (`can_go_pessimistic`)** |
| 紧急矿场turn限制 | `turn < 400` | 收紧回 `turn < 300` |
| `can_go_semi_optimistic` | 不存在 | **新增**（当前v17未使用，预留） |
| `can_go_pessimistic` | 正常 | 包含一段重复代码（bug，不影响运行） |

**v17 最终改动总结：**

1. **Tier 0**：悲观BFS（`can_go_pessimistic`），窄搜索（3列），目标 r+3~r+4，接受 `path_len ≤ 5` 且方向为北向（`dr2 > 0`）的结果
2. **Tier 2**：从乐观BFS改为悲观BFS（`can_go_pessimistic`），减少走入未知区域的概率
3. **紧急矿场**：turn限制收紧回 `< 300`（v16曾放宽到400，v17回退）
4. **JUMP 双重检查**（继承自v15）：landing点不仅需可见，还需路径可达（北2格皆可通行）

**已知问题：**
- `can_go_pessimistic` 函数末尾有两行重复代码（dead code），不影响正确性但应清理。
- `can_go_semi_optimistic` 已定义但未被任何代码引用，属于预留接口。

---

## 测试结论

| 对比 | 胜率 | Reward |
|------|------|--------|
| v15 → v17 | **持平** | **+46 ~ +146** |

v17 在胜率不变的情况下，平均 reward 提升 46~146，主要归因于：
- 悲观BFS减少了工厂走入死胡同的频率
- Tier 0 的北向优先级确保工厂更积极地前进
- 紧急矿场 turn<300 限制避免了后期浪费建造冷却

---

## 各版本架构总览

所有版本（v14-v17）共享相同的基础架构：

```
agent()
  ├── update_state()        — 全局状态更新
  ├── scout_action()         — 侦察/水晶采集
  ├── worker_action()       — 开路/拆墙/水晶采集
  ├── miner_action()         — 去mining node转矿
  └── factory_action()      — 核心决策（建造/导航/JUMP/矿场）
        ├── Mine target selection (ROI-based)
        ├── Urgent mine (adjacent node → BUILD_MINER)
        ├── Mine handling (MOVE/IDLE near friendly mine)
        ├── Navigation (Tier 0/1/2/3/4)
        └── BUILD (during move cooldown)
```

**关键工具函数：**
- `can_go()` — 乐观通行判断（unknown=passable）
- `can_go_pessimistic()` — 悲观通行判断（unknown=wall）
- `bfs_first_step()` — BFS寻路，返回方向+路径长度（v16起）
- `bfs_to_row()` — BFS到目标行
- `bfs_distance()` — BFS计算最短距离
- `calc_mine_roi()` — 矿场投资回报计算
