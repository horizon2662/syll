---
description: "将反复出现的操作模式自动沉淀为可复用 skill"
always: false
---

# 自动沉淀 Skill

当你发现自己在不同任务中重复执行相同的操作序列时，主动将其提炼为 skill。

## 触发条件

满足以下任一条件时，主动提议创建 skill：
1. 同一类型任务执行了 3 次以上（如多次帮用户格式化数据）
2. 用户说"以后遇到这种事就这么做"
3. 一个操作序列超过 5 步且步骤固定
4. 你刚完成一个复杂的调试过程，解决方案可以复用

## 创建流程

1. **确定 skill 名称** — `<动作>-<对象>` 格式，如 `image-resize`, `api-test`, `deploy-check`
2. **创建目录** — `~/.syll/workspace/skills/<skill-name>/SKILL.md`
3. **编写 SKILL.md**：
   - `description` — 一句话说清楚这个 skill 做什么
   - `requires.bins` — 需要的外部命令（如 `["ffmpeg"]`）
   - 适用场景
   - 完整的步骤列表（精确到可以直接执行，不需要猜）
   - 常见变体或参数说明
4. **如果涉及脚本** — 在 `scripts/` 子目录写可执行脚本
5. **测试** — 至少跑一遍确认 skill 能正常工作
6. **告知用户** — 报告创建了什么 skill，在哪里，怎么触发

## SKILL.md 模板

```markdown
---
description: "<一句话描述>"
requires:
  bins: ["<需要的CLI工具>"]
always: false
---

# <Skill 名称>

## 适用场景
- 场景 1
- 场景 2

## 操作步骤

### Step 1: <描述>
具体操作...

### Step 2: <描述>
具体操作...

## 变体
- 如果参数不同怎么做
- 常见注意事项
```

## 注意

- **不要等用户说"保存为 skill"** — 看到重复模式就主动创建
- **创建前确认目录不存在** — 如果已有同名 skill，更新它而不是覆盖
- **给 skill 起能搜到的名字** — 用英文 kebab-case，如 `video-learn` 而不是 `视频学习`
