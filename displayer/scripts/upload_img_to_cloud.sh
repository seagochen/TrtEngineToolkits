#!/bin/bash

# 设置错误处理：脚本在遇到错误时会立即退出
set -e

# 函数：输出错误信息并退出
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# 检查参数数量是否正确
if [ "$#" -ne 3 ]; then
    echo "用法: $0 <device_id> <detect_code> <local_img_file>"
    exit 1
fi

# 获取参数并确保正确引用
DEVICE_ID="$1"
DETECT_CODE="$2"
LOCAL_FILENAME="$3"

# 检查文件是否存在
if [ ! -f "$LOCAL_FILENAME" ]; then
    error_exit "未找到文件: $LOCAL_FILENAME"
fi

# 自动生成远程文件名
REMOTE_FILENAME=$(basename "$LOCAL_FILENAME")

# 连接参数（建议将敏感信息存储在环境变量中）
HOSTNAME="ee-iothub-001.azure-devices.net"
SHARED_ACCESS_KEY="Cxw7v1hcnYYlSzrHXRs07Ppe1LcdP5p8yAIoTL7pcCM="
DETECT_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 获取 SAS URL
echo "获取 SAS URL..."
SAS_URL=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"key":"value"}' \
  "https://azjsfunction006.azurewebsites.net/api/sasurl?deviceid=${DEVICE_ID}&filename=${REMOTE_FILENAME}") || error_exit "无法获取 SAS URL"

echo "SAS URL: $SAS_URL"

# 上传图片文件到远程服务器
echo "上传文件到远程服务器..."
curl -s -X PUT \
  -T "$LOCAL_FILENAME" \
  "$SAS_URL" \
  -H "x-ms-blob-type: BlockBlob" || error_exit "文件上传失败"

echo "文件上传成功"

# 创建消息负载
MSG_TXT=$(cat <<EOF
{
  "local_camera_id": "${DEVICE_ID}",
  "detect_time": "${DETECT_TIME}",
  "detect_code": "${DETECT_CODE}",
  "detect_imgfile": "${REMOTE_FILENAME}"
}
EOF
)

# 生成 SAS 令牌（授权）
# 注意：此处假设 SAS 令牌的生成逻辑由外部服务处理
echo "获取 SAS Token..."
SAS_TOKEN=$(curl -s "https://azjsfunction006.azurewebsites.net/api/index?deviceid=${DEVICE_ID}") || error_exit "无法获取 SAS Token"

echo "SAS Token: ${SAS_TOKEN}"

# 发送消息到 IoT Hub
echo "发送消息到 IoT Hub..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "https://${HOSTNAME}/devices/${DEVICE_ID}/messages/events?api-version=2018-06-30" \
  -H "Authorization: ${SAS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${MSG_TXT}")

if [ "$RESPONSE" -eq 200 ] || [ "$RESPONSE" -eq 204 ]; then
    echo "消息发送成功"
else
    error_exit "消息发送失败，HTTP 状态码: $RESPONSE"
fi
