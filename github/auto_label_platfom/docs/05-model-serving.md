# 05 — 模型服务（SAM2 / SAM3）

## 1. 角色

Model Serving 是 GPU 侧车，对平台只暴露稳定 HTTP（或 gRPC）推理 API。  
标注业务状态（谁拥有 Job、是否采纳）留在 Platform；模型服务无状态或仅缓存编码器特征。

参考源码：

- `github/sam2` — `SAM2ImagePredictor`、`build_sam2_video_predictor`  
- `github/sam3` — `Sam3Processor` 文本提示、video `handle_request` 会话  
- `github/X-AnyLabeling` — ONNX 交互封装与组合模型心智（可选加速路径）

## 2. 部署单元

| 服务 | 职责 | 硬件 |
| --- | --- | --- |
| `sam2-service` | 图像点/框分割；视频传播 | 1×GPU 起 |
| `sam3-service` | 文本/概念分割；视频会话 | 1×GPU 起（可与 SAM2 分卡） |
| `yolo-service`（可选） | 检测预标 | 共享或独立 |
| `vlm-service`（可选） | 任务描述 / 开放词汇辅助 | 大显存 |

平台 API 做鉴权代理与限流；Worker 持有服务账号调用批量接口。

## 3. SAM2 API 契约（逻辑）

### 3.1 图像预测

```http
POST /v1/sam2/image/predict
```

请求要点：

- `image_uri` 或已上传的 `image_id`（服务端拉 S3）  
- `points`: `[{x, y, label: 1|0}]`  
- `boxes`: `[[x0,y0,x1,y1], ...]` 可选  
- `multimask`: bool  

响应：

- `masks`: RLE 或 polygon 列表  
- `scores`: float[]  
- `latency_ms`

实现映射：`set_image` → `predict(...)`。

### 3.2 视频传播

```http
POST /v1/sam2/video/session
POST /v1/sam2/video/session/{id}/prompt
POST /v1/sam2/video/session/{id}/propagate
DELETE /v1/sam2/video/session/{id}
```

- 会话绑定：`episode_id` + `camera_key` + 帧范围（避免整文件超显存）。  
- 长视频：按窗口（如 300 帧）切片传播，平台侧拼接 `track_id`。  
- 实现映射：`init_state` → `add_new_points_or_box` → `propagate_in_video`。

## 4. SAM3 API 契约（逻辑）

### 4.1 图像 + 文本

```http
POST /v1/sam3/image/predict
```

- `image_uri` / `image_id`  
- `text`: 开放词汇概念  
- `exemplars`（可选）：少量框示例  

响应：多实例 `boxes` / `masks` / `scores`。  
实现映射：`set_image` → `set_text_prompt`。

### 4.2 视频会话

对齐官方 request 类型：`start_session` | `add_prompt` | `propagate` | `close_session`。  
平台 Track Worker 负责把结果写成 Prediction 行。

## 5. 与平台 Prediction 协议对齐（Label Studio 启发）

批量预标 Worker 写入统一结构：

```json
{
  "job_id": "...",
  "camera_key": "observation.images.top",
  "frame_index": 12,
  "label": "cube",
  "score": 0.91,
  "geometry": {"type": "mask", "rle": "..."},
  "model_name": "sam3",
  "model_version": "2026.08",
  "prompt": {"type": "text", "text": "cube"}
}
```

Studio 与 Export 只依赖该结构，不直接耦合 PyTorch 类型。

## 6. 预标策略（Project settings）

| 策略 | 行为 |
| --- | --- |
| `sam3_text_keyframes` | 按 fps 抽帧或每 N 帧；对 ontology 中带 `prompt_text` 的标签跑 SAM3 |
| `sam3_then_sam2_track` | 关键帧 SAM3 → 选最高分实例 → SAM2 传播 |
| `interactive_only` | 不批量，仅 Studio 点击 |

阈值：`min_score`、`max_instances_per_label`、相机白名单。

## 7. 性能与工程要点

1. **编码器缓存**：同一 `image_id` 短 TTL 缓存 embedding，支撑连点。  
2. **分辨率**：推理短边 512/1024 可配，坐标映射回原图像素。  
3. **队列隔离**：交互优先级 > 批量预标；批量可低优先级队列。  
4. **多租户公平**：每租户并发槽位；Enterprise 可绑独占 GPU。  
5. **版本钉扎**：`model_version` 写入 Prediction，导出可复现。  
6. **失败降级**：SAM3 不可用时允许仅手动画框 + SAM2 点选。  

## 8. ONNX 可选路径

- 目的：弱 GPU / 边缘节点；与 X-AnyLabeling ONNX 导出对齐。  
- MVP **不要求**；Team 可将 `sam2-service` 换 ONNX Runtime 实现同一 HTTP 契约。  

## 9. YOLO / VLM Adapter（非主路径）

- YOLO：出框 → 可选送 SAM2 精修 mask。  
- VLM：建议 task 文本、失败原因标签；不替代 SAM 做像素级主链路。  
- 均实现同一 Prediction JSON，便于 Studio 无感切换。  

## 10. 安全

- 模型服务只认内网 token；校验 `image_uri` 属于请求租户前缀。  
- 禁止任意 URL fetch（SSRF）；仅允许平台签发的 object key。  
