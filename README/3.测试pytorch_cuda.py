#!/usr/bin/env python3
"""测试 PyTorch 是否能正常使用 CUDA GPU。

用法:
  python README/3.测试pytorch_cuda.py
"""

from __future__ import annotations


def main() -> int:
    print("=" * 50)
    print("PyTorch CUDA GPU 可用性检测")
    print("=" * 50)

    try:
        import torch
    except ImportError as e:
        print(f"[失败] 未安装 PyTorch: {e}")
        print("请先安装，例如:")
        print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        return 1

    print(f"PyTorch 版本     : {torch.__version__}")
    print(f"CUDA 编译版本    : {torch.version.cuda}")
    print(f"cuDNN 版本       : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
    print(f"CUDA 是否可用    : {torch.cuda.is_available()}")
    print(f"GPU 数量         : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print()
        print("[失败] torch.cuda.is_available() == False")
        print("常见排查:")
        print("  1. 驱动: nvidia-smi 能否正常显示 GPU")
        print("  2. 安装的是否为 CPU 版 PyTorch（需安装带 CUDA 的版本）")
        print("  3. PyTorch 的 CUDA 版本需与本机驱动兼容")
        return 1

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"GPU[{i}] 名称      : {props.name}")
        print(f"GPU[{i}] 显存      : {mem_gb:.2f} GB")
        print(f"GPU[{i}] 算力      : {props.major}.{props.minor}")

    print("-" * 50)
    print("执行一次 GPU 张量运算...")
    try:
        device = torch.device("cuda:0")
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = a @ b
        torch.cuda.synchronize()
        print(f"运算结果形状     : {tuple(c.shape)}")
        print(f"结果所在设备     : {c.device}")
        print(f"当前显存占用     : {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")
    except Exception as e:
        print(f"[失败] GPU 运算异常: {e}")
        return 1

    print("-" * 50)
    print("[成功] PyTorch 可以正常使用 CUDA GPU")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
