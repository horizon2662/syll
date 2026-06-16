---
description: "创建新项目或模块的标准流程"
always: false
---

# 项目创建

当用户要求创建新项目、新模块、或初始化开发环境时，按此流程操作。

## 步骤

### 1. 确认需求

- 项目类型：Web 应用、CLI 工具、库、脚本、桌面应用？
- 语言和框架
- 目标目录（默认用 workspace 下的子目录）

### 2. 创建目录结构

标准 Python 项目：
```
project-name/
├── README.md
├── pyproject.toml    # 或 setup.py
├── .gitignore
├── src/
│   └── package_name/
│       ├── __init__.py
│       └── main.py
├── tests/
│   └── test_main.py
└── docs/
```

标准 Node.js 项目：
```
project-name/
├── README.md
├── package.json
├── .gitignore
├── src/
│   └── index.ts
├── tests/
└── tsconfig.json
```

### 3. 初始化

```bash
# Python
cd project-name
git init
# 创建虚拟环境
python -m venv .venv

# Node.js
npm init -y
```

### 4. 编写基础代码

- `README.md` — 项目名、描述、安装步骤、使用方法
- 入口文件 — 最小可运行骨架
- `.gitignore` — 排除 node_modules/, .venv/, __pycache__/, .env 等
- 如果需要敏感配置（API key），创建 `.env.example` 而不是 `.env`

### 5. 验证

- 运行入口文件确认不报错
- 运行测试确认框架正常
- `git add . && git commit -m "init"`

### 6. 报告

告诉用户：
- 项目创建在哪个目录
- 如何运行
- 下一步建议（安装依赖、配置 API key 等）
