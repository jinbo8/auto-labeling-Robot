一条完整 episode 导出（LeRobot Dataset v3）

原始数据集把多条轨迹拼在同一个 mp4 / parquet 里。
本目录是按 episode 切出来的完整一段：视频 + 每一帧关节数据 + 元信息。

dataset: /home/jin/6t/item/auto-labeling-Robot/lerobot/datasets/svla_so100_pickplace
episode_index: 0
task: Pick up the cube and place it in the box.
task_index: 0
fps: 30.0
length: 454 帧  ≈ 15.133 s
全局帧 index: [0, 454)  （左闭右开）
本段 frame_index: 0 .. 453

文件说明:
  README.txt            本说明
  info.json             数据集全局 meta/info.json 的拷贝
  episode_meta.json     该 episode 在 meta/episodes 里的全部字段（含 stats/*）
  task.txt              任务文本（来自 meta/tasks.parquet）
  frames.parquet        该 episode 的全部帧（一行一帧）
  frames.txt            同上，纯文本便于通读
  videos/               从拼接 mp4 按时间戳切开的 top / wrist 片段
  sample_frames/        每个相机的首 / 中 / 末帧 JPG
  sources.json          原始文件路径与裁剪时间戳

视频时间（在原始拼接 mp4 上的秒）:
  observation.images.top: [0.000000, 15.133333] s  src=/home/jin/6t/item/auto-labeling-Robot/lerobot/datasets/svla_so100_pickplace/videos/observation.images.top/chunk-000/file-000.mp4
  observation.images.wrist: [0.000000, 15.133333] s  src=/home/jin/6t/item/auto-labeling-Robot/lerobot/datasets/svla_so100_pickplace/videos/observation.images.wrist/chunk-000/file-000.mp4

对齐关系:
  第 k 帧 (frame_index=k) 的画面 ≈ 切开后视频的 t = k / fps 秒
  parquet 的 timestamp 是相对本 episode 起点的秒，通常等于 frame_index / fps
  action / observation.state 是 6 维关节角（度），顺序见 info.json features.names

