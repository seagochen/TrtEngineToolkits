#!/bin/bash

# 检查是否以 root 用户执行
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

# 提供选择菜单
echo "Jetson Nano の電源モードを選択してください："
echo "1. 全性能モード (MAXN)"
echo "2. 省電力モード (10W)"
read -p "選択 (1/2): " mode

case $mode in
  1)
    echo "全性能モードに切り替えます..."
    sudo nvpmodel -m 0  # 切换到 MAXN 模式
    ;;
  2)
    echo "省電力モードに切り替えます..."
    sudo nvpmodel -m 1  # 切换到 10W 模式
    ;;
  *)
    echo "無効な選択です。スクリプトを終了します。"
    exit 1
    ;;
esac

# System reboot is required for the changes to take effect
echo "変更を有効にするためにシステムを再起動してください。"
read -p "再起動しますか？ (y/n): " reboot

if [[ "$reboot" == "y" ]]; then
    sudo reboot
else
    echo "再起動をキャンセルしました。"
fi
