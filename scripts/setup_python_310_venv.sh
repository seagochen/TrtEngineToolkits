#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
REQ_FILE="${REQUIREMENTS:-requirements.txt}"

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }

if ! command -v python3.10 >/dev/null 2>&1; then
  err "未找到 python3.10，请先运行 install_python310.sh"
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  warn "虚拟环境已存在：$VENV_DIR，跳过创建"
else
  log "创建虚拟环境：$VENV_DIR"
  python3.10 -m venv "$VENV_DIR"
fi

log "升级 pip/setuptools/wheel..."
"$VENV_DIR/bin/pip" install -U pip setuptools wheel

if [[ -f "$REQ_FILE" ]]; then
  log "安装依赖：$REQ_FILE"
  "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
else
  warn "未找到依赖文件：$REQ_FILE，跳过依赖安装"
fi

log "虚拟环境已就绪：$VENV_DIR"
echo "激活方法：source $VENV_DIR/bin/activate"
