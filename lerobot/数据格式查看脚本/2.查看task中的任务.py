"""打印 meta/tasks.parquet 的文件信息、schema、以及每一行内容。"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_PARQUET = (
    REPO_ROOT / "lerobot/datasets/svla_so100_pickplace/meta/tasks.parquet"
)


def _print_section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def show_file_info(path: Path) -> None:
    """文件级信息：路径、体积、parquet schema。"""
    _print_section("1. 文件信息")
    print(f"path: {path}")
    print(f"exists: {path.is_file()}")
    print(f"size_bytes: {path.stat().st_size}")

    table = pq.read_table(path)
    print("-" * 60)
    print("parquet schema:")
    print(table.schema)
    print("-" * 60)
    print(f"num_rows: {table.num_rows}")
    print(f"num_columns: {table.num_columns}")
    print(f"column_names: {table.column_names}")
    meta = table.schema.metadata or {}
    if meta:
        print("-" * 60)
        print("schema metadata keys:", [k.decode() if isinstance(k, bytes) else k for k in meta])
        pandas_meta = meta.get(b"pandas") or meta.get("pandas")
        if pandas_meta:
            text = pandas_meta.decode() if isinstance(pandas_meta, bytes) else pandas_meta
            print("pandas metadata:")
            print(text)


def show_dataframe_info(df: pd.DataFrame) -> None:
    """pandas 读入后的表级信息。"""
    _print_section("2. DataFrame 信息")
    print(f"shape: {df.shape}  (n_tasks={len(df)}, n_cols={df.shape[1]})")
    print(f"columns: {df.columns.tolist()}")
    print(f"index.name: {df.index.name!r}")
    print(f"index.dtype: {df.index.dtype}")
    print("dtypes:")
    print(df.dtypes.to_string())
    print("-" * 60)
    print("完整表格:")
    print(df.to_string())


def show_one_row(df: pd.DataFrame, i: int = 0) -> None:
    """打印一条任务：索引（任务文本）+ 所有列。"""
    _print_section(f"3. 第 {i} 条内容")
    if i < 0 or i >= len(df):
        print(f"越界: i={i}, n={len(df)}")
        return

    row = df.iloc[i]
    index_val = df.index[i]
    print(f"iloc: {i}")
    print(f"index (任务文本): {index_val!r}  type={type(index_val).__name__}")
    print("-" * 60)
    print("所有字段:")
    print(f"  <index>: {index_val}")
    for col, val in row.items():
        print(f"  {col}: {val!r}  type={type(val).__name__}")


def show_all_rows(df: pd.DataFrame) -> None:
    """逐行打印全部任务。"""
    _print_section("4. 全部行")
    print(f"共 {len(df)} 条")
    for i in range(len(df)):
        row = df.iloc[i]
        print("-" * 60)
        print(f"[{i}] index={df.index[i]!r}")
        for col, val in row.items():
            print(f"  {col}: {val}")


def main(row_index: int = 0) -> None:
    path = TASKS_PARQUET
    show_file_info(path)
    df = pd.read_parquet(path)
    show_dataframe_info(df)
    show_one_row(df, i=row_index)
    show_all_rows(df)


if __name__ == "__main__":
    main()
