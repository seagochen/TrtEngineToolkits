#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-detect CUDA & TensorRT install locations and append PATH/LD_LIBRARY_PATH
to the user's shell init file (~/.bashrc by default).

Usage:
  python3 configure_accel_env.py
  python3 configure_accel_env.py --dry-run
  python3 configure_accel_env.py --bashrc ~/.zshrc
"""

import os
import re
import sys
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

# ----------------------------
# Utilities
# ----------------------------
def print_message(msg: str) -> None:
    print(f"\033[1;34m[INFO]\033[0m {msg}")

def print_warn(msg: str) -> None:
    print(f"\033[1;33m[WARN]\033[0m {msg}")

def print_error(msg: str) -> None:
    print(f"\033[1;31m[ERROR]\033[0m {msg}")

def is_jetson() -> bool:
    # Jetson 常见标识文件
    return Path("/etc/nv_tegra_release").exists() or "tegra" in (Path("/proc/device-tree/compatible").read_text(errors="ignore").lower() if Path("/proc/device-tree/compatible").exists() else "")

def which(exe: str) -> Optional[Path]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / exe
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# ----------------------------
# CUDA detection
# ----------------------------
def detect_cuda() -> Optional[Tuple[Path, Path]]:
    """
    Returns (cuda_root, cuda_lib) or None if not found.
    Heuristics:
      1) respect /usr/local/cuda symlink if valid
      2) pick highest version under /usr/local/cuda-*
      3) fallback by locating nvcc in PATH
    """
    candidates: List[Path] = []

    # 1) 常规安装位置
    cuda_symlink = Path("/usr/local/cuda")
    if (cuda_symlink / "bin" / "nvcc").exists():
        candidates.append(cuda_symlink)

    # 2) 版本目录（取最大版本号）
    vers = sorted((Path("/usr/local").glob("cuda-*")), key=lambda p: p.name, reverse=True)
    for v in vers:
        if (v / "bin" / "nvcc").exists():
            candidates.append(v)

    # 3) PATH 里的 nvcc
    nvcc = which("nvcc")
    if nvcc:
        candidates.append(nvcc.parent.parent)  # .../cuda/bin/nvcc -> .../cuda

    # 去重
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))

    for root in uniq:
        lib = root / "lib64"
        if not lib.is_dir():
            lib = root / "lib"
        if (root / "bin" / "nvcc").exists():
            return (root, lib)

    return None

# ----------------------------
# TensorRT detection
# ----------------------------
def _trtexec_ok(root: Path) -> bool:
    """Check if bin/trtexec exists under root (commonly for TRT archives)."""
    return (root / "bin" / "trtexec").exists() and os.access(root / "bin" / "trtexec", os.X_OK)

def _trt_lib_dir(root: Path) -> Optional[Path]:
    """Return plausible TRT lib directory under a given tensorRT root."""
    for sub in ["lib", "lib64", "targets/x86_64-linux-gnu/lib", "targets/aarch64-linux-gnu/lib"]:
        p = root / sub
        if p.is_dir():
            # quick sanity check for core libs
            if list(p.glob("libnvinfer*")):
                return p
    return None

def detect_tensorrt() -> Optional[Tuple[Path, Optional[Path]]]:
    """
    Returns (trt_root, trt_lib_dir) where:
      - trt_root is used to extend PATH with trt_root/bin (if present)
      - trt_lib_dir is used to extend LD_LIBRARY_PATH
    Detection order:
      Jetson:
        - /usr/src/tensorrt (NVIDIA JetPack)
      x86:
        - /opt/tensorrt
        - /usr/local/TensorRT* (highest version first)
        - Fallback: system-wide libs (/usr/lib/*/tensorrt or libnvinfer*)
    """
    j = is_jetson()
    if j:
        root = Path("/usr/src/tensorrt")
        if _trtexec_ok(root) or _trt_lib_dir(root):
            return (root, _trt_lib_dir(root))

    # x86 常见目录
    preferred: List[Path] = []
    opt_trt = Path("/opt/tensorrt")
    if opt_trt.exists():
        preferred.append(opt_trt)

    # 多版本目录（按版本名倒序）
    tdirs = sorted(Path("/usr/local").glob("TensorRT*"), key=lambda p: p.name, reverse=True)
    preferred.extend(tdirs)

    # 去重
    seen = set()
    uniq: List[Path] = []
    for p in preferred:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))

    for root in uniq:
        if _trtexec_ok(root) or _trt_lib_dir(root):
            return (root, _trt_lib_dir(root))

    # 系统库兜底：很多发行版把 TRT 库放在 /usr/lib/x86_64-linux-gnu 或 /usr/lib/aarch64-linux-gnu
    lib_candidates = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu/tensorrt"),
        Path("/usr/lib/aarch64-linux-gnu/tensorrt"),
    ]
    for lc in lib_candidates:
        if lc.is_dir() and list(lc.glob("libnvinfer*")):
            # 没有明确 root，但有库。将 root 置为其父上级，bin 可能不可用。
            return (lc.parent if lc.name in ("tensorrt",) else lc, lc)

    return None

# ----------------------------
# Bashrc editing (idempotent)
# ----------------------------
def inject_block(bashrc_path: Path, marker: str, lines: List[str], dry_run: bool=False) -> None:
    """
    Insert or replace a marked block:
      # >>> {marker}
      <lines...>
      # <<< {marker}
    """
    begin = f"# >>> {marker}"
    end = f"# <<< {marker}"

    if not bashrc_path.exists():
        content = ""
    else:
        content = bashrc_path.read_text()

    block = "\n".join([begin, *lines, end]) + "\n"

    if begin in content and end in content:
        # Replace existing block
        new = re.sub(
            rf"{re.escape(begin)}.*?{re.escape(end)}\n?",
            block,
            content,
            flags=re.DOTALL,
        )
        action = "更新"
    else:
        # Append new block with a preceding newline
        new = content + ("\n" if content and not content.endswith("\n") else "") + block
        action = "追加"

    if dry_run:
        print_message(f"[dry-run] 将在 {bashrc_path} 中{action}标记块：{marker}")
        print(block)
        return

    bashrc_path.write_text(new)
    print_message(f"{bashrc_path} 已{action}标记块：{marker}")

def export_lines(path_bin: Optional[Path], path_lib: Optional[Path]) -> List[str]:
    out: List[str] = []
    if path_bin:
        out.append(f'export PATH="$PATH:{path_bin}"')
    if path_lib:
        out.append('export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}' + f'{path_lib}"')
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Detect CUDA & TensorRT and write PATH/LD_LIBRARY_PATH to shell rc.")
    parser.add_argument("--bashrc", type=str, default=str(Path.home() / ".bashrc"),
                        help="Shell init file to modify (default: ~/.bashrc)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written without modifying files.")
    args = parser.parse_args()

    bashrc_path = Path(args.bashrc)

    # Detect CUDA
    cuda_found = detect_cuda()
    if cuda_found:
        cuda_root, cuda_lib = cuda_found
        cuda_marker = "CUDA Toolkit Environment Variables (auto)"
        cuda_lines = export_lines(cuda_root / "bin", cuda_lib)
        print_message(f"Detected CUDA at: {cuda_root}")
        inject_block(bashrc_path, cuda_marker, cuda_lines, dry_run=args.dry_run)
    else:
        print_warn("CUDA 未检测到；将不会写入 CUDA 相关环境变量。")

    # Detect TensorRT
    trt_found = detect_tensorrt()
    if trt_found:
        trt_root, trt_lib = trt_found
        trt_marker = f"TensorRT Environment Variables (auto, {'Jetson' if is_jetson() else 'x86'})"
        trt_bin = trt_root / "bin" if (trt_root / "bin").is_dir() else None
        trt_lines = export_lines(trt_bin, trt_lib)
        # 额外：若有 trtexec，则提示版本
        trtexec = trt_bin / "trtexec" if trt_bin else None
        if trtexec and trtexec.exists():
            print_message(f"Detected TensorRT: {trt_root} (trtexec: {trtexec})")
        else:
            print_message(f"Detected TensorRT libs at: {trt_lib if trt_lib else trt_root}")
        inject_block(bashrc_path, trt_marker, trt_lines, dry_run=args.dry_run)
    else:
        print_warn("TensorRT 未检测到；将不会写入 TensorRT 相关环境变量。")

    # Epilogue
    if not args.dry_run:
        print("\nIMPORTANT:")
        print(f"  修改已写入：{bashrc_path}")
        print(f"  执行 `source {bashrc_path}` 或重开终端以生效。")
    else:
        print("\n(dry-run) 未对文件做任何修改。")

if __name__ == "__main__":
    main()
