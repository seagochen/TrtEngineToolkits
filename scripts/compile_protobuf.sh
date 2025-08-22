#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# compile_protobuf.sh
# 用法：
#   ./compile_protobuf.sh <PROTOBUF_DIR> <LANG>
#   ./compile_protobuf.sh <LANG>                    # PROTOBUF_DIR 来自环境变量或默认目录
#
# 说明：
#   - LANG 当前支持：python
#   - 输出生成到源码目录结构下(--python_out="$PROTOBUF_DIR")
#   - 需要系统已安装 protoc(setup_all_env.sh 会安装 protobuf-compiler)
# --------------------------

DEFAULT_PROTOBUF_DIR="/opt/SurveillanceService/protobufs/"

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }

if [[ $# -eq 0 ]]; then
  err "Usage: $0 <PROTOBUF_DIR> <LANG>  |  $0 <LANG>"
  exit 1
fi

PROTOBUF_DIR=""
TARGET_LANGUAGE=""

if [[ $# -eq 1 ]]; then
  # 只提供语言。若环境变量 PROTOBUF_DIR 指向有效目录，则优先用它；否则用默认目录
  TARGET_LANGUAGE="$1"
  if [[ -n "${PROTOBUF_DIR:-}" && -d "${PROTOBUF_DIR}" ]]; then
    log "Using PROTOBUF_DIR from environment: ${PROTOBUF_DIR}"
  else
    PROTOBUF_DIR="$DEFAULT_PROTOBUF_DIR"
    log "No directory specified, using default: $PROTOBUF_DIR"
  fi
elif [[ $# -ge 2 ]]; then
  # 同时提供目录和语言
  PROTOBUF_DIR="$1"
  TARGET_LANGUAGE="$2"
fi

if [[ ! -d "$PROTOBUF_DIR" ]]; then
  err "Protobuf directory '$PROTOBUF_DIR' does not exist."
  exit 1
fi

if ! command -v protoc >/dev/null 2>&1; then
  err "protoc not found. Please install protobuf-compiler (apt) first."
  exit 1
fi

log "Protobuf root: $PROTOBUF_DIR"
log "Target language: $TARGET_LANGUAGE"

# 递归查找 .proto
mapfile -t PROTO_FILES < <(find "$PROTOBUF_DIR" -type f -name "*.proto")
if [[ ${#PROTO_FILES[@]} -eq 0 ]]; then
  warn "No .proto files found under $PROTOBUF_DIR. Nothing to do."
  exit 0
fi

case "$TARGET_LANGUAGE" in
  python|py)
    log "Compiling to Python..."
    # 为了在包中正确 import，最好在各级目录放 movement_fiters.py(可选)
    # 这里仅做编译输出到源码目录
    for proto in "${PROTO_FILES[@]}"; do
      proto_dir="$(dirname "$proto")"
      set -x
      protoc -I "$PROTOBUF_DIR" --python_out="$PROTOBUF_DIR" "$proto"
      set +x
    done
    log "Protobuf -> Python generated under $PROTOBUF_DIR"
    ;;

  *)
    err "Unsupported language: $TARGET_LANGUAGE (supported: python)"
    exit 2
    ;;
esac

log "Protobuf compilation finished."
