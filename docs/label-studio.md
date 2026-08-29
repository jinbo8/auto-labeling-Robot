# Label Studio 环境配置与使用说明

源码：`github/label-studio`  
上游：https://github.com/HumanSignal/label-studio  
官方文档：https://labelstud.io/guide/

Label Studio 是通用网页标注工具，同一套界面可以标图像、文本、音频、视频、时序、HTML。标签界面用 XML 模板描述，适合自定义任务和接自己的模型做预标注。

当前克隆包版本：`1.24.0.dev0`，需要 **Python 3.10+**（`<4`）。

## 1. 它能做什么

- 多用户、多项目；标注结果绑定账号
- 内置大量模板：目标检测、分割、NER、ASR、分类等
- 从本地文件、S3、GCS、Azure 导入
- 导出 JSON，再用 SDK/Converter 转成 COCO、YOLO、VOC、Conll 等
- 通过 [ML Backend](https://github.com/HumanSignal/label-studio-ml-backend) 做预标注、在线学习、主动学习

和 CVAT 的差别：CVAT 更偏计算机视觉产线（视频、3D、质检）；Label Studio 更偏「什么数据都能配一套界面」。

## 2. 选哪种安装方式

| 方式 | 适用 | 数据存储 | 入口 |
| --- | --- | --- | --- |
| **pip / 本仓库 poetry** | 单机试用、开发 | 默认 SQLite | http://localhost:8080 |
| **官方 Docker 单容器** | 简单部署 | 挂目录里的 SQLite | http://localhost:8080 |
| **docker compose** | 更接近生产 | PostgreSQL + Nginx | http://localhost （映射 8080→8085） |

本机已经有完整源码，推荐两种：

- 先验证功能：pip 或 poetry（SQLite）
- 长期用：在 `github/label-studio` 里 `docker compose up`

**注意：默认端口 8080 与 CVAT 冲突，不要两套同时用 8080。**

## 3. 方式 A：pip 最快试用

适合确认「能不能打开页面」。不依赖本仓库源码。

```bash
python3 --version   # 需要 >= 3.10
python3 -m venv ~/.venvs/label-studio
source ~/.venvs/label-studio/bin/activate
pip install -U pip
pip install label-studio
label-studio
```

浏览器打开 http://localhost:8080 ，按提示注册第一个账号（该账号即为管理员）。

数据默认写在用户目录下的 Label Studio 数据目录。停掉用 Ctrl+C。

指定端口避免和 CVAT 抢 8080：

```bash
label-studio start --port 8085
```

## 4. 方式 B：用本仓库源码开发/运行

适合要改代码或跑最新 nightly。

### 步骤 1：环境

```bash
cd /home/jin/6t/item/auto-labeling-Robot/github/label-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip poetry
poetry install
```

若只要跑服务、不改前端，这样即可。改 React 前端还需要 [Bun](https://bun.sh/)，见 `web/README.md`。

### 步骤 2：迁移并启动

```bash
python label_studio/manage.py migrate
python label_studio/manage.py collectstatic --noinput
python label_studio/manage.py runserver 0.0.0.0:8085
```

或用仓库 Makefile（SQLite + debug）：

```bash
make migrate-dev
make run-dev
```

`run-dev` 默认仍是 8080，若 CVAT 占用了，改成：

```bash
DJANGO_DB=sqlite DEBUG=true DJANGO_SETTINGS_MODULE=core.settings.label_studio \
  uv run python label_studio/manage.py runserver 0.0.0.0:8085
```

打开 http://localhost:8085 注册账号。

## 5. 方式 C：Docker Compose（PostgreSQL）

```bash
cd /home/jin/6t/item/auto-labeling-Robot/github/label-studio
docker compose up -d
```

组成：

- `app`：uWSGI 跑 Label Studio
- `db`：PostgreSQL 17（`pgautoupgrade/pgautoupgrade`）
- `nginx`：静态资源与反代

端口（见 `docker-compose.yml`）：

- **8080 → 容器 8085**（主站）
- **8081 → 容器 8086**

数据：

- 标注与上传文件：`github/label-studio/mydata`
- 数据库：`github/label-studio/postgres-data`

第一次构建镜像较慢。构建本地镜像：

```bash
docker build -t heartexlabs/label-studio:latest .
```

只跑官方镜像、不经过 compose：

```bash
mkdir -p /home/jin/6t/item/auto-labeling-Robot/autolabel/data/label-studio
docker run --rm -it -p 8085:8080 \
  -v /home/jin/6t/item/auto-labeling-Robot/autolabel/data/label-studio:/label-studio/data \
  heartexlabs/label-studio:latest
```

这里把宿主机端口改成 8085，避免和 CVAT 冲突。

带 MinIO（本地模拟 S3）：

```bash
docker compose -f docker-compose.yml -f docker-compose.minio.yml up -d
```

## 6. 第一次标注流程

1. 打开页面并登录。
2. **Create Project**，填写名称。
3. **Labeling Setup**：选模板，例如 Computer Vision → Object Detection with Bounding Boxes。  
   把 `<Label value="person"/>` 改成你的类别。界面由 XML 驱动，可直接改配置。
4. **Import**：上传图片、zip，或填 JSON 任务列表。每条任务至少要有可访问的数据 URL 或已上传文件。
5. 进入 **Label**，画框/选类别，点 Submit。
6. **Export**：默认 JSON。需要 COCO/YOLO 时在导出格式列表里选，或事后用 [label-studio-sdk converter](https://github.com/HumanSignal/label-studio-sdk)。

检测模板核心片段类似：

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person"/>
    <Label value="car"/>
  </RectangleLabels>
</View>
```

`$image` 对应导入数据里的字段名。

## 7. 接自己的模型（预标注）

1. 另开一个 [label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend) 服务，例如本地 `http://localhost:9090`。
2. 项目 Settings → **Model** → 填 ML Backend URL。
3. 打开 **Use predictions for pre-labeling**。新任务会带模型框，人工再改。

本仓库的 `scripts/vllm.py` 是 vLLM / OpenAI 兼容接口示例，可以包一层 ML Backend，把 VLM 预测写回 Label Studio，而不是直接替 Label Studio 启动。

## 8. 常用命令与配置

```bash
# 指定数据目录
export LABEL_STUDIO_BASE_DATA_DIR=/home/jin/6t/item/auto-labeling-Robot/autolabel/data/label-studio
label-studio

# 公网/反向代理时告诉它对外 URL
export LABEL_STUDIO_HOST=http://your-domain:8085
```

Compose 里同样可用 `LABEL_STUDIO_HOST`。

停止 compose：

```bash
docker compose down
```

数据库在 `postgres-data/`，删这个目录等于清空项目。

## 9. 常见问题

**和 CVAT 抢 8080**  
先 `docker compose -f github/cvat/docker-compose.yml down`，或 Label Studio 改用 8085。

**poetry install 失败**  
`pyproject.toml` 里 `label-studio-sdk` 从 GitHub 固定 commit 拉取，需要能访问 GitHub。可改用 `pip install label-studio` 装发行版。

**导入的图片不显示**  
本地文件必须通过 Label Studio 上传或可被浏览器访问的 URL。直接填 `/home/...` 路径通常不行。Docker 部署时文件要在挂载的 `mydata` 里。

**Windows 上 lxml 编译失败**  
用官方 wheel 或 WSL2。本机是 Linux，一般无此问题。

**前端改了页面没变化**  
源码运行时要在 `web/` 里 `bun install && bun run build`（或 `make frontend-dev` 热更新）。

## 10. 和本仓库的关系

适合多模态或自定义表单式标注。视觉检测若只求速度，优先 X-AnyLabeling；要审核流和视频时间轴，优先 CVAT。导出结果建议落到 `autolabel/data/labeled/`。
