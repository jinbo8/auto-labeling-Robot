from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

DEMO_DIR = Path(__file__).resolve().parent
DATA_PARQUET = "lerobot/datasets/svla_so100_pickplace/data/chunk-000/file-000.parquet"
EPISODES_PARQUET = (
    "lerobot/datasets/svla_so100_pickplace/meta/episodes/chunk-000/file-000.parquet"
)

# 跳过每条 episode 里的 stats/* 明细，先看索引与视频对齐信息
EPISODE_META_COLS = [
    "episode_index",
    "length",
    "tasks",
    "dataset_from_index",
    "dataset_to_index",
    "data/chunk_index",
    "data/file_index",
    "videos/observation.images.top/chunk_index",
    "videos/observation.images.top/file_index",
    "videos/observation.images.top/from_timestamp",
    "videos/observation.images.top/to_timestamp",
    "videos/observation.images.wrist/chunk_index",
    "videos/observation.images.wrist/file_index",
    "videos/observation.images.wrist/from_timestamp",
    "videos/observation.images.wrist/to_timestamp",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
]


def show_row():
    # 查看某一行数据，一行代表一个样本,时间计算方式为帧号/fps
    df = pd.read_parquet(DATA_PARQUET)
    print(df.shape)
    print(df.columns.tolist())

    # 整行数据全部打印（第 0 行）
    row = df.iloc[19231]
    for col, val in row.items():
        print(f"{col}: {val}")


def show_episodes(episode_index: int = 0):
    """读取 meta/episodes，查看某条轨迹的索引与视频时间范围。"""
    df = pd.read_parquet(EPISODES_PARQUET)
    print("=" * 60)
    print("episodes meta")
    print("=" * 60)
    print(f"shape: {df.shape}  (n_episodes={len(df)}, n_cols={df.shape[1]})")
    print(f"episode_index range: {df.episode_index.min()} .. {df.episode_index.max()}")
    print(f"all columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    row = df.loc[df.episode_index == episode_index].iloc[0]
    print("-" * 60)
    print(f"episode_index={episode_index} 核心字段:")
    for col in EPISODE_META_COLS:
        print(f"  {col}: {row[col]}")

    # 全局帧 index 落在 [from, to) 内
    print("-" * 60)
    print(
        f"数据帧范围: index ∈ [{row['dataset_from_index']}, {row['dataset_to_index']})  "
        f"共 {row['length']} 帧"
    )
    print(
        f"top 视频时间: "
        f"[{row['videos/observation.images.top/from_timestamp']:.3f}, "
        f"{row['videos/observation.images.top/to_timestamp']:.3f}] s"
    )
    print(
        f"wrist 视频时间: "
        f"[{row['videos/observation.images.wrist/from_timestamp']:.3f}, "
        f"{row['videos/observation.images.wrist/to_timestamp']:.3f}] s"
    )


def dump_all_to_txt(
    out_name: str = "svla_so100_pickplace_dump.txt",
    include_episode_stats: bool = False,
):
    """把 episodes 元数据 + 全部帧数据写入 lerobot/demo 下的 txt，方便通读。"""
    out_path = DEMO_DIR / out_name
    episodes = pd.read_parquet(EPISODES_PARQUET)
    frames = pd.read_parquet(DATA_PARQUET)

    ep_cols = (
        list(episodes.columns)
        if include_episode_stats
        else [c for c in EPISODE_META_COLS if c in episodes.columns]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# svla_so100_pickplace 数据导出\n")
        f.write(f"# episodes: {EPISODES_PARQUET}\n")
        f.write(f"# frames:   {DATA_PARQUET}\n\n")

        f.write("=" * 80 + "\n")
        f.write("1. EPISODES META\n")
        f.write("=" * 80 + "\n")
        f.write(f"shape: {episodes.shape}\n")
        f.write(f"columns: {ep_cols}\n\n")
        for i, row in episodes.iterrows():
            f.write(f"--- episode_index={row['episode_index']} ---\n")
            for col in ep_cols:
                f.write(f"  {col}: {row[col]}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("2. ALL FRAMES (parquet 每一行)\n")
        f.write("=" * 80 + "\n")
        f.write(f"shape: {frames.shape}\n")
        f.write(f"columns: {frames.columns.tolist()}\n\n")
        for i, row in frames.iterrows():
            f.write(f"--- iloc={i} index={row.get('index', i)} ---\n")
            for col, val in row.items():
                f.write(f"  {col}: {val}\n")
            f.write("\n")

    print(f"已写入: {out_path}")
    print(f"  episodes: {len(episodes)} 条")
    print(f"  frames:   {len(frames)} 行")
    return out_path


if __name__ == "__main__":
    show_episodes(episode_index=1)
    dump_all_to_txt()
