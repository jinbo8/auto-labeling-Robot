# X-AnyLabeling 环境配置与使用说明

源码：`github/X-AnyLabeling`  
上游：https://github.com/CVHub520/X-AnyLabeling  
版本：v4.0.5  
许可：GPL-3.0

X-AnyLabeling 是跨平台**桌面**标注软件（PyQt6），不是网页服务。单机标图、内置大量检测/分割/姿态/OCR/SAM 模型、导出 YOLO/COCO/VOC，是本仓库里上手最快的自动标注入口。

## 1. 它能做什么

- 图像 / 视频标注：矩形、旋转框、多边形、掩码、点、线、立方体等
- 任务：分类、检测、实例分割、姿态、OBB、跟踪、OCR、深度、Grounding、VLM 等
- 推理引擎：ONNX Runtime（CPU/GPU）、也可接远程 vLLM / SGLang（见 [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server)）
- 导入导出：COCO、VOC、YOLO、DOTA、MOT、MASK、PPOCR、MMGD、VLM-R1、ShareGPT 等

需要图形界面（本机桌面或转发 X11/Wayland）。SSH 纯终端无法弹出窗口。

## 2. 环境要求

| 项 | 要求 |
| --- | --- |
| Python | **3.11–3.13**，官方推荐 **3.12** |
| 系统 | Linux / Windows / macOS |
| GUI | PyQt6，Linux 需可用的桌面会话 |
| GPU（可选） | NVIDIA + 匹配的 CUDA / cuDNN，用于 ONNX Runtime GPU |
| 磁盘 | 模型下载到 `~/xanylabeling_data/`，SAM/YOLO 权重可达数百 MB～数 GB |

同一环境里 **cpu / gpu / gpu-cu11 / gpu-cu13 只能装一个**，也不要同时装 `onnxruntime` 和 `onnxruntime-gpu`。

CUDA 与依赖对应关系（摘自项目 `docs/zh_cn/get_started.md`）：

| CUDA | pip extra | onnxruntime-gpu | cuDNN |
| --- | --- | --- | --- |
| 无 / CPU | `cpu` | （CPU 包） | — |
| 11.x | `gpu-cu11` | `>=1.15,<1.19` | 8.x |
| 12.x（默认 GPU） | `gpu` | `>=1.18.1,<1.27` | 9.x |
| 13.x | `gpu-cu13` | `>=1.27,<1.28` | 9.x |

先看本机 CUDA：

```bash
nvidia-smi
```

没有独显或暂时不用 GPU，走 CPU 即可。

## 3. 一步一步安装（推荐：源码可编辑安装）

源码已在 `github/X-AnyLabeling`，不要再 clone 一份到别的路径后忘了重装（editable 安装会绑当前目录）。

### 步骤 1：建 Python 3.12 环境

Conda：

```bash
conda create -n x-anylabeling-cu12 python=3.12 -y
conda activate x-anylabeling-cu12
```

无 Conda 时用 venv：

```bash
python3.12 -m venv /home/jin/6t/item/auto-labeling-Robot/.venv-xal
source /home/jin/6t/item/auto-labeling-Robot/.venv-xal/bin/activate
```

CPU 环境把名字里的 `cu12` 换成 `cpu` 即可，后面安装 extra 改成 `cpu`。

### 步骤 2：卸掉可能冲突的旧包

```bash
pip uninstall -y anylabeling x-anylabeling-cvhub onnxruntime onnxruntime-gpu || true
```

### 步骤 3：安装本仓库源码

```bash
cd /home/jin/6t/item/auto-labeling-Robot/github/X-AnyLabeling
pip install -U uv

# GPU，CUDA 12.x（本机有 NVIDIA 时优先这条）
uv pip install -e ".[gpu]"

# 仅 CPU
# uv pip install -e ".[cpu]"

# CUDA 11.x
# uv pip install -e ".[gpu-cu11]"
```

二次开发再加 `dev`：`uv pip install -e ".[gpu,dev]"`。

### 步骤 4：验证

```bash
xanylabeling version
xanylabeling checks
xanylabeling config    # 打印配置文件路径
```

`checks` 会打印系统、Qt、ONNX Runtime 提供器。GPU 安装成功时应能看到 `CUDAExecutionProvider`。

### 步骤 5：启动 GUI

```bash
xanylabeling
```

常用启动方式：

```bash
# 直接打开一个图片目录
xanylabeling /path/to/images

# 标注 json 写到指定目录，且不把图片 base64 塞进 json
xanylabeling /path/to/images --output /path/to/labels --nodata --autosave

# 预设类别
xanylabeling /path/to/images --labels person,car,dog

# Fedora KDE 鼠标发飘时可试
# xanylabeling --qt-platform xcb
```

完整参数：`xanylabeling --help`。

## 4. 第一次标注流程

1. 启动后把界面语言改为简体中文：菜单 **帮助 → 语言**。
2. **文件 → 打开目录**（Ctrl+U），选图片文件夹；或 Ctrl+I 打开单张图、Ctrl+O 打开视频。
3. 左侧选矩形 / 多边形等工具，框选目标，输入或选择类别。
4. `A` / `D` 上一张 / 下一张。Ctrl+S 保存。默认在图片同目录写同名 `.json`（LabelMe 风格）。
5. 打开 **自动保存**（`--autosave` 或设置里），避免漏存。

导出给训练用：

- 菜单里对应 **YOLO / VOC / COCO / DOTA / MASK / MOT / PPOCR** 等导出
- 或命令行：`xanylabeling convert` 查看任务，例如 `xanylabeling convert xlabel2yolo --help`

建议把原始图放在 `autolabel/data/raw/`，导出标签放到 `autolabel/data/labeled/`。

## 5. 使用内置模型自动标注

1. 快捷键 **Ctrl+A**（或左侧 AI 按钮）打开模型面板。
2. 下拉选择模型，例如 YOLOv8 检测、SAM 分割。第一次会下载权重到：
   - 默认：`~/xanylabeling_data/models/<模型名>/`
   - 若加了 `--work-dir`，则在该工作目录下
3. 等加载完成后，对当前图运行自动标注；需要批量时用 **Ctrl+B** 批量预测。
4. 人工改错框，再保存。

国内下载 GitHub Release 失败时：

- 设置里把模型下载源改成 ModelScope（用户手册「7.7 模型下载源配置」）
- 或按 [模型列表](../github/X-AnyLabeling/docs/zh_cn/model_zoo.md) 手动下 onnx，再「加载自定义模型」

加载自定义 YOLO 的要点：

1. 把 `pt` 导出成 **固定尺寸** 的 `onnx`（动态轴需确认适配器支持）。
2. 复制 `anylabeling/configs/auto_labeling/` 下对应 yaml（如 `yolov5s.yaml`）。
3. 改 `model_path` 为本地路径，改 `classes` 为你的类别。
4. **不要改 `type` 字段**。在界面里加载该 yaml。

## 6. 远程大模型 / vLLM（可选）

桌面端可以把部分 VLM 任务指到远程 OpenAI 兼容接口。本仓库 `scripts/vllm.py` 记录了 vLLM 与 DashScope 的调用方式，可与 X-AnyLabeling 的 Chatbot / VQA / 远程推理配合，而不是替代 GUI。

独立远程服务项目：[X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server)。多人共享 GPU 时走 Server，本机 GUI 只负责改框。

本机起一个 OpenAI 兼容服务的示意（模型路径按实际修改）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name Qwen2-VL \
  --max-model-len 2048
```

然后在 X-AnyLabeling 的 Chatbot / 对应模型配置里填写 `http://127.0.0.1:8000/v1`。细节见项目文档 `docs/zh_cn/chatbot.md`、`docs/zh_cn/vqa.md`。

## 7. 不装源码：pip 或绿色包

仅要稳定发行版：

```bash
uv pip install "x-anylabeling-cvhub[gpu]"   # 或 [cpu]
xanylabeling
```

GUI 安装包：https://github.com/CVHub520/X-AnyLabeling/releases  
解压即用，但功能往往落后源码，出问题也难查。GPU 包必须和本机 CUDA 一致，否则会静默回退 CPU。

## 8. 常用快捷键

| 快捷键 | 作用 |
| --- | --- |
| Ctrl+U | 打开图片目录 |
| Ctrl+S | 保存 |
| A / D | 上一张 / 下一张 |
| Ctrl+A | 自动标注面板 |
| Ctrl+B | 批量标注 |
| Ctrl+J | 绘制 / 编辑模式 |
| Ctrl+Z | 撤销 |
| Delete | 删选中对象 |
| Ctrl+G | 标注统计 |

更多见源码 `docs/zh_cn/user_guide.md`。

## 9. 常见问题

**`xanylabeling: command not found`**  
虚拟环境没激活，或没装到当前环境。用 `which xanylabeling` 确认。

**能启动但 GPU 没用上**  
`xanylabeling checks` 看 Provider。确认只装了 `onnxruntime-gpu`、CUDA/cuDNN 大版本匹配。Linux 还需要 `libcudnn` 能被找到。

**模型下载卡住**  
换 ModelScope，或手动下载后改 yaml 的 `model_path`。

**Qt 报 image allocation limit**  
大图启动加：`--qt-image-allocation-limit 2048`（单位 MB）。

**和旧 AnyLabeling 混装**  
必须先 `pip uninstall anylabeling`。本包名是 `x-anylabeling-cvhub`，入口仍是 `xanylabeling`。

**在服务器上没界面**  
这是桌面程序。无显示器时请用 CVAT / Label Studio，或配 X11 转发 / 远程桌面。

## 10. 和本仓库的关系

本仓库自动标注流水线优先建议：

1. 用 X-AnyLabeling 对 `autolabel/data/raw/` 做检测/分割预标 + 人工修正
2. 导出 YOLO 或 COCO 到 `autolabel/data/labeled/`
3. 需要多人审核或视频任务时，把数据再导入 CVAT

项目自带用户手册、模型 zoo、CLI 转换说明都在：

- `github/X-AnyLabeling/docs/zh_cn/get_started.md`
- `github/X-AnyLabeling/docs/zh_cn/user_guide.md`
- `github/X-AnyLabeling/docs/zh_cn/cli.md`
