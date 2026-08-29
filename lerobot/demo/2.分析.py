import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _patch_multiprocess  # noqa: F401  # Python 3.12.0 + multiprocess 退出报错

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parent / "datasets" / "svla_so100_pickplace"

ds = LeRobotDataset(
    "lerobot/svla_so100_pickplace",
    root=ROOT,
)
print(ds.meta.info.codebase_version, ds.num_episodes, ds[0]["task"])
print(ds[0]["action"].shape, ds[0]["observation.images.top"].shape)
