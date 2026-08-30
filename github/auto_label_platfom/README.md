# Auto Label Platform — 代码与方案

面向具身智能数据团队的 Web 自动标注平台（**机器人 / VLA V1**）。

本目录同时包含：

- **方案文档**：`docs/`、`openapi/sketch.yaml`
- **可运行代码**：按 roadmap 工程分解落在下方 monorepo

## 快速开始（本地 MVP）

```bash
cd github/auto_label_platfom

# 1) Python 依赖（建议用已有 conda 环境）
pip install -e packages/lerobot_index -e services/api -e services/worker
pip install -r requirements.txt

# 2) 初始化 DB + seed 用户，导入样例集并拆 Job
export ALP_DATA_ROOT="$(pwd)/../../lerobot/datasets/svla_so100_pickplace"
python -m alp_api.cli init
python -m alp_api.cli seed-demo

# 3) 启动 API（默认 http://127.0.0.1:8000）
uvicorn alp_api.main:app --reload --app-dir services/api/src

# 4) （另开终端）前端
cd apps/web && npm install && npm run dev
```

默认账号（seed）：

| 邮箱 | 密码 | 角色 |
|------|------|------|
| manager@local | manager123 | manager |
| annotator@local | annotator123 | annotator |
| reviewer@local | reviewer123 | reviewer |

API 文档：http://127.0.0.1:8000/docs

## 代码目录

```
auto_label_platfom/
  docs/                 # 产品与架构方案
  openapi/              # API 草图
  packages/
    lerobot_index/      # LeRobot v3 索引 / sidecar / COCO
  services/
    api/                # FastAPI 领域服务
    worker/             # QA / 预标 / 导出任务
    sam2/               # SAM2 HTTP（MVP stub，可接真权重）
    sam3/               # SAM3 HTTP（MVP stub）
  apps/
    web/                # Portal + Studio（React + Vite）
  deploy/
    compose/            # Docker Compose
  requirements.txt
  Makefile
```

## 已定决策（摘要）

| 项 | 决策 |
| --- | --- |
| 形态 | Web 平台（非 fork X-AnyLabeling 桌面） |
| 垂直 | 机器人 / VLA；LeRobot v3 一级公民 |
| 模型 | SAM2/SAM3 主链路；MVP 可 stub |
| 栈 | FastAPI + SQLAlchemy(SQLite 本地 / Postgres Compose) + React |

详细见 [docs/01-product-vision.md](docs/01-product-vision.md) 起各章。
