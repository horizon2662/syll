# Syll 长程重构路线图（论文 + 架构方案 合并）

> 目标：把 enhanced 框架从"零件齐全但未串联"升级为**一个统一的长程执行系统**——
> 主/sub 通信清晰、每技能独立记忆、长程可恢复。不重写旧代码，全部用继承/组合新增。

## 第一部分：痛点诊断（对准代码）

| 痛点 | 现状定位 | 根因 |
|---|---|---|
| 通信混乱 | `subagent.py:_announce_result` 把整段结果塞 system 消息；中间状态主 agent 完全失明 | 单向终态、fire-and-forget |
| 两套 subagent | `SubagentManager`(fire-and-forget) vs `GUIExecuteSubAgent`(同步单步) | 无统一抽象与 I/O 契约 |
| 失败不重规划 | `_failure_counts` 超限只静默吞掉 | 无 Diagnose→Align 环 |
| 无 per-skill 记忆 | subagent 一次性、`max_iterations=15` | 记忆只在全局 MemoryStore |
| 记忆无检索 | `StructuredMemory` 单一线性历史 | 无分层、无按需取 |
| 长程不可续 | 无 checkpoint/resume | plan 状态不持久化 |

## 第二部分：论文 → 改造点 映射

| 改造点 | 论文/出处 | 借鉴的核心机制 |
|---|---|---|
| **Fold（折叠回传）** | Context-Folding (arXiv:2510.11967, ICML'26) | branch 开子轨迹→return 只回传摘要，中间步丢弃。`max_session` 限预算。仓库 `agents/fold_agent.py` 的 `process_item` |
| **文件系统合同** | Anthropic 多agent研究系统 | subagent 产物写文件、只回传引用 → 消除传话游戏 |
| **I/O 契约** | Anthropic | objective+output_format+tools_hint+boundaries 必须齐 |
| **依赖查询** | TaskWeave (arXiv:2606.01199) | 每子任务声明 Ψ：执行前先解析上游结果 |
| **Diagnose→Align** | TaskWeave FPDA | 失败回灌诊断、主 agent 重规划（非静默） |
| **四角色闭环** | Mobile-Agent-v3 (arXiv:2602.16855) | Manager/Worker/Reflector/Notetaker；状态 X_t=(指令,截图,子目标,反馈,笔记) |
| **Notetaker 门槛写** | Mobile-Agent-v3 | 仅成功才 `N_{t+1}=u_C`；失败不污染记忆 |
| **分层上下文压缩** | Mobile-Agent-v3 | 最近 N 轮全保留 + 更早拼接 action conclusion 摘要 |
| **procedural memory** | Mem0 (arXiv:2504.19413) | episodic/semantic 之外的"怎么做"=技能；multi-scope(user/agent/run/org) |
| **记忆自组织链接** | A-Mem (arXiv:2502.12110) | note construction + link generation + memory evolution |
| **checkpoint/resume** | Anthropic context eng / Orchestration 综述 | compaction + 结构化笔记 + State 单元显式管理 |
| **文件系统即记忆** | Letta Filesystem | 74% LoCoMo 打败专门系统 → 暂不上向量库 |

## 第三部分：分阶段路线图

### Phase 1 — 通信层重构（最高优先，治"混乱"）
新增 `contract.py` + `blackboard.py`，重构 `subagent.py` → `unified_subagent.py`。
- [ ] `SubagentContract` / `SubagentResult` 数据契约
- [ ] `Blackboard`：每 run 一个 `workspace/agents/{run_id}/`（contract.json / progress.md 心跳 / result.json / artifacts/）
- [ ] `ReturnTool`：fold 原语，subagent 只回传摘要 + 引用
- [ ] `UnifiedSubagentManager`：三模式（background/step/skill），`_announce` 改为只发摘要+引用
- **验证**：spawn 后 `workspace/agents/{id}/progress.md` 有心跳；主 agent 收到的 system 消息只有摘要+引用，不是整段输出。
- **不动**：旧 `SubagentManager.spawn()` 签名保留（`spawn(task, label, ...)` 仍可用），平滑迁移。

### Phase 2 — 记忆分层（治"无 per-skill 记忆"）
新增 `skill_memory.py`。
- [ ] `SkillMemory`：每技能 `workspace/skills/{skill}/SKILL.md`（procedural）+ `system_prompt.md`
- [ ] `ingest(lessons, status_ok)` 门槛写入（Notetaker 规则 + A-Mem 去重链接）
- [ ] `context_for_subagent()`：spawn 时把技能记忆切片注入 `contract.context_slice`
- [ ] 主 agent 维护 `PROJECT.md`（全局目标/约束），手动或半自动蒸馏进技能
- **验证**：同一技能第二次 spawn 时能读到上次积累的 lesson；失败尝试不写入。

### Phase 3 — 长程：分层 plan + checkpoint（治"不可续"）
新增 `hierarchical_plan_manager.py`（继承 `PlanManager`）。
- [ ] milestone→step 两层（主 agent 产里程碑，subagent 展开步骤）
- [ ] `checkpoint(plan)` / `resume()`：把 plan 状态存 `session_state.json`
- [ ] 失败 → `diagnosis` 触发主 agent 重规划（消费 Phase 1 的 diagnosis 字段）
- [ ] （可选）对话级 compaction：主 agent 上下文逼近上限时总结+重开，对接现有 `StructuredMemory.compress_history`
- **验证**：长任务中断后 `resume()` 能从最后 checkpoint 继续。

### Phase 4 — 收敛 GUI subagent（治"两套抽象"）
把 `GUIExecuteSubAgent` 改造成 `UnifiedSubagentManager` 的 `mode="step"`：
- Worker=execute_step，Reflector=ActionVerifier，Notetaker=SkillMemory，Manager=主 agent
- 这样 GUI 的 TVAE 闭环和通用 subagent 共用同一套合同/黑板/记忆。

### Phase 5 — （可选）向量化检索
当 SKILL.md / history 膨胀到检索不灵时，再上 Mem0 multi-signal（语义+BM25+entity）。
按 Letta 数据，文件系统在 LoCoMo 已 74%，不急。

## 第四部分：文件清单（本目录）

```
refactor_skeleton/
├── ROADMAP.md                      # 本文件
├── contract.py                     # SubagentContract / SubagentResult（I/O 契约）
├── blackboard.py                   # Blackboard（文件系统共享态）
├── skill_memory.py                 # SkillMemory（per-skill procedural 记忆 + 门槛写）
├── unified_subagent.py             # UnifiedSubagentManager（替代 subagent.py，三模式+fold）
└── hierarchical_plan_manager.py    # HierarchicalPlanManager（继承 plan_manager，milestone+checkpoint）
```

## 关键设计取舍

1. **fold 用文件系统，不用内存拷贝**。FoldAgent 是在内存里 copy history 做 branch（训练场景，同进程）。Syll 是多进程/跨会话的生产场景——用文件系统黑板上通信更稳、可恢复、天然支持并行 subagent。本质都是"中间步丢弃、只回摘要"。
2. **保留 MessageBus 唤醒，但不塞内容**。主 agent 仍靠 `publish_inbound` 的轻量通知得知 subagent 完成，但通知里只有摘要+引用；完整产物在黑板上按需读。这是 Anthropic 思路与现有总线的折中。
3. **门槛写记忆是可靠性的关键**。不加门槛，SKILL.md 会变成噪声垃圾桶——这正是 Mobile-Agent-v3 Notetaker `N_{t+1}` 条件更新的意义。
4. **procedural memory 用 markdown，暂不上向量库**。Letta 实测文件系统已够强；等膨胀再上 Mem0。
