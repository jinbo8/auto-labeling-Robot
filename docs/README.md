# 标注工具说明文档

本目录对应仓库 `github/` 下已克隆的三个开源标注项目。三套工具都能做视觉数据标注，但定位不同，**不要同时占用 8080 端口**。

| 项目 | 形态 | 默认入口 | 适合场景 | 文档 |
| --- | --- | --- | --- | --- |
| CVAT | Docker 网页服务 | http://localhost:8080 | 团队协作、视频/3D、任务分发与质检 | [cvat.md](./cvat.md) |
| Label Studio | pip / Docker 网页服务 | http://localhost:8080 | 多模态（图/文/音/视频）、模板灵活、接 ML Backend | [label-studio.md](./label-studio.md) |
| X-AnyLabeling | 本机桌面应用 | `xanylabeling` | 单机快速标图、内置 YOLO/SAM 自动标注、导出 YOLO/COCO | [x-anylabeling.md](./x-anylabeling.md) |

源码位置（已 gitignore，不会推到本仓库）：

```
github/cvat
github/label-studio
github/X-AnyLabeling
```

当前克隆版本：

- CVAT：develop（约 v2.44.3 之后）
- Label Studio：nightly（包版本 `1.24.0.dev0`）
- X-AnyLabeling：`v4.0.5`

建议起步顺序：单人先跑 **X-AnyLabeling**；需要多人协作或视频任务再上 **CVAT**；需要文本/音频/自定义界面再上 **Label Studio**。
