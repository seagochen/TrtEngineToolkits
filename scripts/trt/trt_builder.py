#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys

def check_requirements():
    """检查 trtexec 是否存在"""
    if not shutil.which("trtexec"):
        print("Error: trtexec 未安装或不在 PATH 中", file=sys.stderr)
        sys.exit(1)

def validate_onnx(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX 文件不存在: {path}")
    if not path.endswith(".onnx"):
        raise ValueError(f"文件不是 ONNX 模型: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"ONNX 文件不可读: {path}")

def validate_engine_path(path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        print(f"创建目录: {directory}")
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"目录不可写: {directory}")

def build_engine(model_cfg):
    name = os.path.basename(model_cfg["onnx_path"])
    print(f"\n=== Processing model: {name} ===")

    onnx_path   = model_cfg["onnx_path"]
    engine_path = model_cfg["engine_path"]
    precision   = model_cfg.get("precision", "").lower()
    min_shapes  = model_cfg.get("min_shapes")
    opt_shapes  = model_cfg.get("opt_shapes")
    max_shapes  = model_cfg.get("max_shapes")
    verbose     = model_cfg.get("verbose", False)

    # 验证文件和路径
    validate_onnx(onnx_path)
    validate_engine_path(engine_path)

    # 如果提供了 input_tensors / output_tensors，就简单打印信息
    for inp in model_cfg.get("input_tensors", []):
        print(f"  输入张量: {inp['name']} 维度{inp['dimensions']}")
    for out in model_cfg.get("output_tensors", []):
        print(f"  输出张量: {out['name']} 维度{out['dimensions']}")

    # 构造 trtexec 命令
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}"
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    if min_shapes:
        cmd.append(f"--minShapes={min_shapes}")
    if opt_shapes:
        cmd.append(f"--optShapes={opt_shapes}")
    if max_shapes:
        cmd.append(f"--maxShapes={max_shapes}")
    if verbose:
        cmd.append("--verbose")

    print("执行命令:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print(f"[✔] 成功构建: {engine_path}")
    else:
        print(f"[✖] 构建失败: {name}", file=sys.stderr)
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TRT engine via trtexec")
    parser.add_argument("config", help="配置文件路径（JSON）")
    args = parser.parse_args()

    if not os.path.isfile(args.config) or not os.access(args.config, os.R_OK):
        print(f"Error: 配置文件不可用: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # 检查 trtexec
    import shutil
    check_requirements()

    models = cfg.get("models", [])
    if not models:
        print("Error: 没有在配置中找到任何 model 条目", file=sys.stderr)
        sys.exit(1)

    for model in models:
        build_engine(model)

if __name__ == "__main__":
    main()
