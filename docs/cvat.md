# CVAT 环境配置与使用说明

源码：`github/cvat`  
上游：https://github.com/cvat-ai/cvat  
官方文档：https://docs.cvat.ai/docs/

CVAT（Computer Vision Annotation Tool）是自托管的网页标注平台，适合图像、视频、点云，以及多人协作、任务分发、审核。本仓库已克隆源码，推荐用 Docker Compose 启动，不要从源码直接 `manage.py runserver`。

## 1. 它能做什么

- 标注图像、视频、3D 点云：框、多边形、掩码、关键点、立方体、标签等
- 按 Project → Task → Job 组织数据，分配标注员
- 导入/导出 COCO、YOLO、VOC、KITTI、MOT、CVAT XML 等 20+ 格式
- 通过 Nuclio 接入 SAM、YOLOv7 等模型做自动/半自动标注
- Python SDK / CLI / REST API 可嵌入流水线

浏览器请用 Chrome 或 Edge。Firefox 可能有问题，Safari 不支持。

## 2. 环境要求

| 项 | 建议 |
| --- | --- |
| 系统 | Linux x86_64（本机已满足） |
| Docker Engine | 已安装并可用 `docker` 命令（建议把当前用户加入 `docker` 组） |
| Docker Compose | `docker compose` 插件（v2） |
| 内存 | 至少 8 GB，视频任务建议 16 GB+ |
| 磁盘 | 镜像约数 GB，数据卷随数据集增长 |
| GPU | 基础标注不需要；自动标注 GPU 函数需要 NVIDIA Container Toolkit |
| 端口 | **8080**（Web）、8090（Traefik 内部）。与 Label Studio 冲突 |

检查：

```bash
docker --version
docker compose version
docker info
```

若提示 permission denied，执行后重新登录：

```bash
sudo usermod -aG docker "$USER"
```

## 3. 一步一步启动

以下命令都在本仓库根目录执行。源码已经在 `github/cvat`，不必再 clone。

### 步骤 1：进入目录

```bash
cd /home/jin/6t/item/auto-labeling-Robot/github/cvat
```

### 步骤 2：（可选）指定访问地址

本机浏览器访问可跳过。局域网其他机器访问时：

```bash
export CVAT_HOST=本机IP或域名
```

### 步骤 3：拉起默认栈

当前目录是 develop 分支，默认会拉 `cvat/server:dev`、`cvat/ui:dev`。第一次会下载较多镜像，需要能访问 Docker Hub。

```bash
docker compose up -d
```

固定到某个正式版（更稳）：

```bash
export CVAT_VERSION=v2.44.3
docker compose up -d
```

查看状态：

```bash
docker compose ps
```

相关容器包括：`cvat_server`、`cvat_ui`、`cvat_db`、`cvat_redis_inmem`、`cvat_redis_ondisk`、`traefik`、ClickHouse / Grafana 等。全部 `running` 后再打开页面。

### 步骤 4：创建超级管理员

网页上自己注册的账号默认几乎没有权限，必须先建 superuser：

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

按提示输入用户名、邮箱、密码。

### 步骤 5：登录

浏览器打开 http://localhost:8080 ，用上一步账号登录。

## 4. 第一次标注流程

1. **Projects** → 创建项目，填写名称。
2. 在项目里定义标签（例如 `person`、`car`），或稍后在 Task 里定义。
3. **Tasks** → **Create new task**：
   - 填任务名
   - 选择/新建标签
   - 上传图片、视频，或挂云存储
   - 提交后等待服务端解包、抽帧
4. 打开任务 → 进入 Job，使用左侧工具画框/多边形。
5. 保存（Ctrl+S）。完成后把 Job 状态改为 completed。
6. 在 Task 页 **Export dataset**，选 COCO / YOLO / VOC 等格式下载。

常用快捷键（标注页）：

- `N` 下一帧，`P` 上一帧
- `F` 下一物体，`D` 删除
- `Ctrl+S` 保存

## 5. 开启自动标注（可选）

默认栈没有 Nuclio，模型不会出现在 UI 里。

### 5.1 带 serverless 重新启动

不要只执行 `docker compose up`。先停再带 overlay 启动：

```bash
cd /home/jin/6t/item/auto-labeling-Robot/github/cvat
docker compose down
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

本仓库对应 Nuclio Dashboard 镜像为 `quay.io/nuclio/dashboard:1.16.3-amd64`。`nuctl` 版本必须与之匹配。

### 5.2 安装 nuctl

```bash
NUCLIO_VER=1.16.3
wget "https://github.com/nuclio/nuclio/releases/download/${NUCLIO_VER}/nuctl-${NUCLIO_VER}-linux-amd64"
chmod +x "nuctl-${NUCLIO_VER}-linux-amd64"
sudo ln -sf "$(pwd)/nuctl-${NUCLIO_VER}-linux-amd64" /usr/local/bin/nuctl
nuctl version
```

### 5.3 部署模型函数

CVAT 起来之后再部署，例如 SAM 和 YOLOv7（CPU）：

```bash
./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio
./serverless/deploy_cpu.sh serverless/onnx/WongKinYiu/yolov7/nuclio
```

查看：

```bash
nuctl get function --platform local
```

GPU 函数需要先装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，再用 `./serverless/deploy_gpu.sh`。

部署成功后，在 Task 的 **Actions → Automatic annotation** 里能看到模型。

## 6. 常用运维

```bash
# 看日志
docker compose logs -f cvat_server

# 停止（数据在 named volume 里，不会丢）
docker compose down

# 停止并删数据卷（不可恢复）
docker compose down -v

# 更新镜像后重启
docker compose pull
docker compose up -d
```

数据保存在 Docker volume：`cvat_db`、`cvat_data`、`cvat_keys` 等。备份重点是 `cvat_db` 和 `cvat_data`。

局域网访问：

```bash
export CVAT_HOST=192.168.x.x
docker compose up -d
```

HTTPS 见同目录 `docker-compose.https.yml`。

## 7. SDK / CLI（可选）

```bash
pip install cvat-sdk cvat-cli
```

可在脚本里创建任务、上传数据、导出标注。REST 文档在登录后的 `/api/docs`。

## 8. 常见问题

**页面打不开 / 502**  
等 1–2 分钟让 `cvat_server` 完成 migrate。`docker compose ps` 看是否有 Exited。

**8080 被占用**  
Label Studio 默认也占 8080。先停另一套，或改 CVAT 的 Traefik 端口映射（`docker-compose.yml` 里 `traefik.ports`）。

**注册了账号但看不到任务**  
普通用户默认无权限。用 superuser 登录 Django Admin（`/admin`），把用户加入 `user` 等组。

**国内拉镜像慢**  
给 Docker 配镜像加速，或预先 `docker pull cvat/server:dev cvat/ui:dev postgres:15-alpine redis:7.2.11-alpine traefik:v3.6`。

**自动标注函数部署失败**  
确认 serverless compose 已 up、`nuctl` 大版本与 Dashboard 一致、函数加在 `cvat_cvat` 网络上（`deploy_cpu.sh` 已写 `--platform-config`）。

## 9. 和本仓库的关系

适合作为团队标注后台。导出 YOLO/COCO 后可放到 `autolabel/data/labeled/`。单机、只要检测框时，用 X-AnyLabeling 更快。
