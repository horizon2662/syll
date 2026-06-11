---
description: "在代码库中搜索和导航的技巧"
always: false
---

# 代码搜索与导航

当你需要理解一个陌生代码库或找到某段代码时，按此策略操作。

## 从入口点开始

不要随机翻文件。先找到入口：
- Web 应用：`app.py`, `main.py`, `server.js`, `index.ts`
- 库/包：`__init__.py`, `index.js`, `mod.rs`
- CLI 工具：`cli.py`, `commands/`, `bin/`
- 看项目的 `README.md`, `pyproject.toml`, `package.json` 了解结构

## 搜索策略

### 按功能搜索
```bash
# 搜索函数/类定义
grep -rn "def function_name" --include="*.py"
grep -rn "class ClassName" --include="*.py"

# 搜索所有引用
grep -rn "variable_name" --include="*.py"
```

### 按模式搜索
- 找 API 路由：`grep -rn "route\|endpoint\|@app\." --include="*.py"`
- 找配置读取：`grep -rn "config\|settings\|env\|getenv" --include="*.py"`
- 找数据库操作：`grep -rn "query\|execute\|insert\|update" --include="*.py"`
- 找错误处理：`grep -rn "except\|try:\|catch\|error" --include="*.py"`

### 找最近改动
```bash
git log --oneline -20          # 最近 20 条提交
git diff HEAD~5 --nameonly     # 最近 5 次提交改了哪些文件
git blame file.py -L 10,20     # 某段代码是谁什么时候写的
```

## 理解代码流程

找到入口后，跟着调用链走：
1. 读入口函数
2. 找到它调用的核心函数 → 读取
3. 重复直到理解完整流程
4. 用 exec 跑一个小测试验证理解

## 导航技巧

- `list_dir` 列出目录结构
- `read_file` 读具体文件
- `exec` 运行 `grep`, `find`, `git` 等命令做精确搜索
- 多个独立的搜索可以并行执行
