# GUI Enhanced Branch — 完整代码对比与架构说明

> 分支：`gui-enhanced`（基于 `main` / syll v0.2.0）
> 目标：在**不破坏原始代码**的前提下，通过继承/组合/新增模块增强 GUI 自动化能力

---

## 目录

1. [总览：分支改动范围](#1-总览分支改动范围)
2. [架构设计原则](#2-架构设计原则)
3. [Phase 1：TVAE 验证系统](#3-phase-1tvae-验证系统)
4. [Phase 2：结构化计划与记忆](#4-phase-2结构化计划与记忆)
5. [Phase 3：GUI 执行子代理](#5-phase-3gui-执行子代理)
6. [Phase 4：语义轨迹与空间分析](#6-phase-4语义轨迹与空间分析)
7. [GUI 执行监控浮层](#7-gui-执行监控浮层)
8. [UI-TARS 工具增强（直接替换）](#8-ui-tars-工具增强直接替换)
9. [新增 Skills](#9-新增-skills)
10. [新增 VideoLearn 工具](#10-新增-videolearn-工具)
11. [后向兼容与开关控制](#11-后向兼容与开关控制)
12. [文件清单](#12-文件清单)

---

## 1. 总览：分支改动范围

### 文件统计

| 类型 | 数量 | 说明 |
|------|------|------|
| 新增 Python 模块 | 14 | enhanced/ 目录 11 个 + gui_monitor + enhanced_ui_tars + video_learn |
| 新增模板文件 | 6 | prompt_templates (2) + prompt_deltas (4) |
| 新增 Skills | 7 | code-edit, debug, search-navigate 等 |
| **替换**的原始文件 | 1 | `ui_tars.py`（改动过大，无法通过继承实现） |
| 修改的原始文件 | **0** | 除 ui_tars.py 外，原始代码零修改 |

### 分支对比图

```
main (原始 syll v0.2.0)
 └── gui-enhanced
      ├── syll/agent/aloha/act/enhanced/     ← 全新目录
      ├── syll/desktop/gui_monitor.py        ← 全新文件
      ├── syll/agent/tools/ui_tars.py        ← 替换（增强版）
      ├── syll/agent/tools/video_learn.py    ← 全新文件
      └── syll/skills/{7个新skill}/          ← 全新目录
```

---

## 2. 架构设计原则

### 2.1 继承优先，修改禁止

所有增强模块都遵循以下规则：

| 增强方式 | 适用场景 | 示例 |
|----------|----------|------|
| **继承** (Inheritance) | 需要覆盖父类方法 | `VerifiedPlanner(AlohaPlanner)`, `StructuredMemory(MemoryStore)`, `EnhancedAlohaPlannerTool(AlohaPlannerTool)` |
| **组合** (Composition) | 需要包装/增强，不适合继承 | `EnhancedTraceGenerator` 包装 `TraceGenerator`, `GUIExecuteSubAgent` 组合多个组件 |
| **独立模块** (Standalone) | 完全独立的新功能 | `ActionVerifier`, `PlanManager`, `SpatialAnalyzer`, `GuiMonitor` |
| **直接替换** | 改动遍布文件内部，继承代价过高 | `ui_tars.py`（详见第8节） |

### 2.2 配置驱动，一键开关

```python
# 全部开启（默认）
cfg = EnhancedConfig()

# 全部关闭（回退到原始行为）
cfg = EnhancedConfig.all_disabled()

# 按需开关
cfg = EnhancedConfig(enable_tvae_verification=True, enable_gui_subagent=False)
```

---

## 3. Phase 1：TVAE 验证系统

> 参考：VeriGUI (Baidu, 2026) — Think-Verify-Action-Expectation (TVAE) 框架

### 3.1 ActionVerifier（新增，独立模块）

**文件**：`syll/agent/aloha/act/enhanced/action_verifier.py`

**原始代码**：无对应模块

**功能**：验证 GUI 操作是否真正改变了屏幕

```python
# 两种验证模式
verifier = ActionVerifier(pixel_diff_threshold=0.005)

# 模式 1：像素差异（快速，零 LLM 成本）
result = verifier.verify_pixel_diff(before_b64, after_b64)
# → VerifyResult(status=SUCCESS/NO_CHANGE/UNCERTAIN, confidence, pixel_diff_score)

# 模式 2：LLM 期望验证（准确，消耗 token）
result = await verifier.verify_with_expectation(after_b64, expectation, model="gpt-4o")
```

**关键设计**：
- `VerifyResult` 数据类包含 `status`、`confidence`、`pixel_diff_score`、`diagnosis`
- 像素差异使用 numpy 灰度对比，忽略 <10 灰度级的噪声
- LLM 验证使用 litellm，支持任意模型

### 3.2 VerifiedPlanner（继承 AlohaPlanner）

**文件**：`syll/agent/aloha/act/enhanced/verified_planner.py`

**原始代码**：`syll/agent/aloha/act/planner.py` → `AlohaPlanner`

**继承关系**：
```
AlohaPlanner                    # 原始，不修改
  └── VerifiedPlanner           # 增强，继承
        - 覆盖 plan()          # 添加 TVAE 参数
        - 覆盖 _get_system_prompt()  # 使用 TVAE 系统提示
        + _inject_verification_feedback()  # 新方法
        + _load_prompt_delta()  # 新方法
        + recover()             # 新方法
```

**对比原始 `AlohaPlanner.plan()` 签名**：
```python
# 原始（4 个参数）
async def plan(self, task, guidance_trajectory="", screenshot_b64="", action_history=None)

# 增强（7 个参数，全部可选，后向兼容）
async def plan(self, task, guidance_trajectory="", screenshot_b64="", action_history=None,
               previous_verify_result=None,   # ← TVAE 验证反馈
               spatial_context="",            # ← 空间分析上下文
               action_type_hint="",           # ← 动作类型提示
               prompt_delta="")               # ← 自定义提示增量
```

**新增的 prompt 模板**：

| 文件 | 用途 |
|------|------|
| `prompt_templates/verified_planner/system.txt` | TVAE 系统提示（包含 Expectation 规范、自修复规则） |
| `prompt_templates/verified_planner/user.txt` | 增强的用户提示模板 |
| `prompt_deltas/click_delta.txt` | Click 动作专用提示 |
| `prompt_deltas/drag_delta.txt` | Drag 动作专用提示 |
| `prompt_deltas/type_delta.txt` | Type 动作专用提示 |
| `prompt_deltas/scroll_delta.txt` | Scroll 动作专用提示 |

**TVAE 系统提示 vs 原始系统提示**：

```diff
 # 原始 planner/system.txt
 You are a helpful planning assistant...
 Always output JSON with: Observation, Reasoning, Current Step, Action, Expectation.

+# TVAE enhanced planner/system.txt
+### TVAE Verification Protocol (Enhanced)
+You are operating under a Think–Verify–Action–Expectation (TVAE) framework.
+Your actions are verified against the actual screen state after execution.
+
+#### Your Expectation Field is CRITICAL
+- ✅ Good: "The File menu will expand, showing options: New, Open, Save, Exit"
+- ❌ Bad: "Something will change"
+
+#### When Previous Action Failed (NO_CHANGE detected)
+1. STOP and diagnose
+2. Try a DIFFERENT approach
+3. NEVER repeat the exact same action
+
+#### Recovery Strategies
+- Click missed? → keyboard navigation
+- Element not visible? → scroll first
+- Dialog blocking? → dismiss it
```

---

## 4. Phase 2：结构化计划与记忆

> 参考：MGA (WSDM'25) — 多代理架构中的结构化记忆

### 4.1 PlanManager（新增，独立模块）

**文件**：`syll/agent/aloha/act/enhanced/plan_manager.py`

**原始代码**：无对应模块

**功能**：将执行计划持久化为 Markdown 文件，支持 Plan Agent 和 Execute Sub-Agent 之间通过文件系统共享状态

```python
pm = PlanManager(workspace=Path("~/.syll/workspace"))
plan = pm.create_plan("my_skill", "Open Chrome and search X", ["Step 1", "Step 2"])
pm.save_plan(plan)
pm.update_step(plan, step_index=1, status="DONE", result="OK")
step = pm.get_current_step(plan)
```

**Markdown 格式示例**：
```markdown
# Execution Plan: my_skill
**Task**: Open Chrome and search X

## Steps
### 🔵 Step 1: Open Chrome
- **Action**: click(500, 30)
- **Expectation**: Chrome window opens

### ⬜ Step 2: Type in search bar
```

### 4.2 StructuredMemory（继承 MemoryStore）

**文件**：`syll/agent/aloha/act/enhanced/structured_memory.py`

**原始代码**：`syll/agent/memory.py` → `MemoryStore`

**继承关系**：
```
MemoryStore                     # 原始，不修改
  └── StructuredMemory          # 增强，继承
        + record_step()          # 新方法：记录验证结果
        + get_execution_context()  # 新方法：替代原始 action_history
        + compress_history()     # 新方法：LLM 压缩
```

**关键改进**：

| 原始 MemoryStore | StructuredMemory |
|------------------|------------------|
| 存储原始 action 字符串 | 存储 (action, expectation, verify_status, diagnosis) |
| 上下文随步数线性增长 | 压缩旧步骤 + 保留最近 N 步 |
| 无错误模式识别 | 自动聚类重复错误 |

**`get_execution_context()` 替代 `action_history`**：
```python
# 返回压缩上下文，大幅减少 token 消耗
context = mem.get_execution_context(max_recent_steps=3)
# → "## Previous Steps (summary)\n...\n## Recent Steps\n✅ Step 5: Click File → SUCCESS\n..."
```

---

## 5. Phase 3：GUI 执行子代理

> 参考：CoACT-1 (OSWorld SOTA 56.4%) — 编排器委托给 GUI 操作员

### 5.1 GUIExecuteSubAgent（新增，组合模式）

**文件**：`syll/agent/aloha/act/enhanced/gui_execute_subagent.py`

**原始代码**：无对应模块

**功能**：封装独立的 TVAE 执行循环，不污染主代理的上下文窗口

```
GUIExecuteSubAgent
  ├── planner: VerifiedPlanner | AlohaPlanner    # 组合
  ├── executor: AlohaExecutor                     # 组合
  ├── verifier: ActionVerifier                    # 组合
  └── plan_manager: PlanManager                   # 组合
```

**执行流程**：
```
execute_step(plan, step_index)
  1. 读取步骤描述
  2. 截取 before 截图
  3. Planner → 生成 action + expectation
  4. Executor → 执行 action
  5. 等待 → 截取 after 截图
  6. ActionVerifier → 像素差异验证
  7. 如果 NO_CHANGE → 重试（带恢复提示）
  8. 返回 StepResult
```

---

## 6. Phase 4：语义轨迹与空间分析

### 6.1 SpatialAnalyzer（新增，独立模块）

**文件**：`syll/agent/aloha/act/enhanced/spatial_analyzer.py`

**原始代码**：无对应模块

**功能**：用 VLM 分析截图，生成结构化 UI 描述，注入 Planner 的提示中

```python
analyzer = SpatialAnalyzer(model="gpt-4o")
description = await analyzer.analyze(screenshot_b64)
# → "## Layout\n菜单栏在顶部...\n## Interactive Elements\n- Button: 'File' (top-left)..."
```

### 6.2 EnhancedTraceGenerator（组合 TraceGenerator）

**文件**：`syll/agent/aloha/act/enhanced/enhanced_trace_generator.py`

**原始代码**：`syll/agent/aloha/learn/trace_generator.py` → `TraceGenerator`

**组合关系**（非继承）：
```python
class EnhancedTraceGenerator:
    def __init__(self, base_generator: TraceGenerator, spatial_analyzer: SpatialAnalyzer | None):
        self.base_generator = base_generator      # 组合原始生成器
        self.spatial_analyzer = spatial_analyzer   # 可选空间分析
```

**增强内容**：

| 原始 TraceGenerator | EnhancedTraceGenerator |
|---------------------|------------------------|
| 仅 Observation/Think/Action/Expectation | + `semantic_action`: "Click the File menu" |
| 无动作分类 | + `action_type`: click/drag/type/scroll/hotkey |
| 无 UI 布局描述 | + `spatial_context`: 结构化 UI 描述 |

---

## 7. GUI 执行监控浮层

**文件**：`syll/desktop/gui_monitor.py`（新增）

**原始代码**：`syll/desktop/` 目录下仅有 `ghost.py` 和 `__init__.py`

**功能**：PyQt6 实时进度浮层

```
┌─────────────────────────────────────────┐
│  👻  ⚡ GUI 自动化执行中                  │
│      打开 Chrome 访问 google.com         │
│      ████████░░░  3 / 8                  │
│      ▸ click(start='(960, 540)')        │
│      ⚠ 请勿移动鼠标或操作键盘             │
└─────────────────────────────────────────┘
```

**关键设计**：
- **进程隔离**：作为独立子进程运行 `python -m syll.desktop.gui_monitor --serve`，避免 Qt 主线程问题
- **状态文件通信**：`~/.syll/.gui_monitor_state.json`（JSON 轮询，300ms）
- **配置开关**：`config.json` 中 `tools.gui.monitor: true/false`
- **Ghost 状态映射**：idle → ghost-idle-follow.svg, working → ghost-working-thinking.svg, error → ghost-gui-help.svg

**API（供 agent 调用，无需 Qt 依赖）**：
```python
from syll.desktop.gui_monitor import write_gui_state, launch_gui_monitor, stop_gui_monitor

write_gui_state(status="running", instruction="...", step=3, max_steps=8)
launch_gui_monitor()   # 启动子进程
stop_gui_monitor()     # 停止并清理
```

---

## 8. UI-TARS 工具增强（直接替换）

**文件**：`syll/agent/tools/ui_tars.py`（**替换**原始文件）

### ⚠️ 为什么不能继承？

原始 `ui_tars.py` 的改动遍布文件内部，涉及：
- `Conversation` 数据类新增 `img_size` 字段
- `_compute_model_size()` 完全重写（自适应分辨率）
- `_call_uitars()` 内部维度注入
- `_execute_action()` 动作解析简化
- `execute()` 主循环集成 monitor

这些改动互相依赖，无法通过覆盖少量方法实现。**直接替换是最务实的方案。**

### 增强版 vs 原始版对比

#### 8.1 自适应分辨率

```python
# 原始：硬编码 XGA/WXGA 目标
def _compute_model_size(w, h):
    from syll.agent.tools.coordinate_transform import SCALING_TARGETS
    for tw, th in SCALING_TARGETS.values():
        if abs(tw / th - ratio) < 0.02 and tw < w:
            return tw, th
    return (1280, 800)

# 增强：自适应分层，保留更多细节
_RESOLUTION_TIERS = {
    "16:9": [(3840,2160), (2560,1440), (1920,1080), ...],
    "16:10": [(2560,1600), (1920,1200), ...],
    "4:3": [(2048,1536), ...],
    "3:2": [(2256,1504), ...],
}
def _compute_model_size(w, h):
    # 支持 config.json 覆盖：auto|original|1440p|WxH
    # 4K 屏幕 → 2560×1440 (44%像素) vs 原始 1280×800 (12%像素)
```

#### 8.2 截图维度注入

```python
# 原始：截图不告诉模型图片多大
messages.append({"role": "user", "content": [{"type": "image_url", ...}]})

# 增强：注入维度信息
content_parts = [
    {"type": "image_url", ...},
    {"type": "text", "text": f"[Screenshot size: {iw}x{ih} pixels. Coordinates must be within (0,0)-({iw},{ih}).]"},
]
```

#### 8.3 GUI Monitor 集成

```python
# 原始：无进度反馈
for step in range(1, steps + 1):
    ...

# 增强：每步更新浮层
self._monitor_launch()
for step in range(1, steps + 1):
    self._monitor_write(status="running", instruction=..., step=step, max_steps=steps)
    ...
self._monitor_write(status="finished", ...)
```

#### 8.4 动作解析简化

```python
# 原始：支持 UI-TARS v1 <point>x y</point> 格式 + v1.5 (x,y) 格式
def _parse_coords(s):
    m = re.search(r"<point>(\d+)\s+(\d+)</point>", s)  # v1
    m = re.search(r"\((\d+)\s*,\s*(\d+)\)", s)          # v1.5

# 增强：仅 v1.5 格式，更简洁
def _parse_action_args(args_str):
    for m in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
        args[m.group(1)] = m.group(2)
```

---

## 9. 新增 Skills

| Skill | 文件 | 功能 |
|-------|------|------|
| `code-edit` | `skills/code-edit/SKILL.md` | 代码编辑工作流：读取→分析→编辑→验证→记录 |
| `debug` | `skills/debug/SKILL.md` | 系统性调试：复现→定位→分析根因→修复→验证 |
| `search-navigate` | `skills/search-navigate/SKILL.md` | 代码库搜索导航：入口点→grep→调用链 |
| `project-setup` | `skills/project-setup/SKILL.md` | 项目创建：确认需求→目录结构→初始化→验证 |
| `code-review` | `skills/code-review/SKILL.md` | 代码审查：正确性→安全性→错误处理→风格→性能 |
| `auto-skill` | `skills/auto-skill/SKILL.md` | 自动沉淀 skill：检测重复模式→创建 SKILL.md |
| `video-learn` | `skills/video-learn/SKILL.md` | 视频示教学习：下载→转录→分析→生成 skill |

这些 skills 是**纯 Markdown** 文件，不修改任何 Python 代码，仅作为 agent 的行为指导。

---

## 10. 新增 VideoLearn 工具

**文件**：`syll/agent/tools/video_learn.py`（新增）

**功能**：从在线视频自动学习桌面操作技能

**流水线**：
```
video_learn(task="Premiere Pro add subtitles")
  → Brave Search 搜索教程视频
  → yt-dlp 下载
  → VideoAnalyzer 两阶段分析（快扫+精扫）
  → 生成 SKILL.md + 截图
  → 写入 memory 笔记
```

---

## 11. 后向兼容与开关控制

### 11.1 EnhancedConfig 一键控制

```python
from syll.agent.aloha.act.enhanced.config import EnhancedConfig

# 默认：全部开启
cfg = EnhancedConfig()

# 一键关闭：回到原始行为
cfg = EnhancedConfig.all_disabled()

# 按阶段开关
cfg = EnhancedConfig(
    enable_tvae_verification=True,    # Phase 1: TVAE 验证
    enable_verified_planner=True,     # Phase 1: 验证感知规划器
    enable_prompt_delta=True,         # Phase 1: 动作类型提示
    enable_plan_persistence=True,     # Phase 2: 计划持久化
    enable_structured_memory=True,    # Phase 2: 结构化记忆
    enable_gui_subagent=True,         # Phase 3: 执行子代理
    enable_semantic_trace=True,       # Phase 4: 语义轨迹
    enable_spatial_context=True,      # Phase 4: 空间分析
)
```

### 11.2 UI-TARS Monitor 控制

```json
// config.json
{
  "tools": {
    "gui": {
      "monitor": true,              // GUI monitor 开关（默认 true）
      "modelResolution": "auto"     // auto | original | 1440p | 1080p | 2560x1440
    }
  }
}
```

### 11.3 零侵入保证

| 模块 | 原始文件是否修改 | 回退方式 |
|------|-----------------|----------|
| enhanced/ 全部模块 | ❌ 不修改 | `EnhancedConfig.all_disabled()` |
| gui_monitor.py | ❌ 新增文件 | `config.json: tools.gui.monitor = false` |
| ui_tars.py | ⚠️ 替换 | `git checkout main -- syll/agent/tools/ui_tars.py` |
| video_learn.py | ❌ 新增文件 | 不注册即可 |
| 7 个新 skills | ❌ 新增文件 | 删除目录即可 |

---

## 12. 文件清单

### 新增文件（`gui-enhanced` 分支独有）

```
syll/agent/aloha/act/enhanced/
├── __init__.py
├── config.py                          # EnhancedConfig 开关
├── action_verifier.py                 # TVAE 动作验证
├── verified_planner.py                # 继承 AlohaPlanner
├── plan_manager.py                    # 计划持久化
├── structured_memory.py               # 继承 MemoryStore
├── gui_execute_subagent.py            # 执行子代理
├── spatial_analyzer.py                # 空间分析
├── enhanced_trace_generator.py        # 语义轨迹
├── enhanced_planner_tool.py           # 继承 AlohaPlannerTool
├── verify.py                          # 验证测试脚本
├── prompt_templates/
│   └── verified_planner/
│       ├── system.txt
│       └── user.txt
└── prompt_deltas/
    ├── click_delta.txt
    ├── drag_delta.txt
    ├── type_delta.txt
    └── scroll_delta.txt

syll/desktop/gui_monitor.py            # GUI 进度浮层
syll/agent/tools/video_learn.py        # 视频学习工具

syll/skills/
├── code-edit/SKILL.md
├── debug/SKILL.md
├── search-navigate/SKILL.md
├── project-setup/SKILL.md
├── code-review/SKILL.md
├── auto-skill/SKILL.md
└── video-learn/SKILL.md
```

### 替换文件（与 main 不同）

```
syll/agent/tools/ui_tars.py            # 增强版（含 monitor + 自适应分辨率）
```

### 未修改的原始文件（确认一致）

```
syll/agent/aloha/act/planner.py        ✅ 一致
syll/agent/aloha/act/executor.py       ✅ 一致
syll/agent/aloha/act/trajectory_manager.py  ✅ 一致
syll/agent/aloha/learn/trace_generator.py   ✅ 一致
syll/agent/aloha/learn/log_processor.py     ✅ 一致
syll/agent/aloha/learn/screenshot_processor.py  ✅ 一致
syll/agent/memory.py                   ✅ 一致
syll/agent/gui_click.py                ✅ 一致
syll/agent/gui_skill.py                ✅ 一致
syll/skills/gui-agent/SKILL.md         ✅ 一致
syll/skills/README.md                  ✅ 一致
```
