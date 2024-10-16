#!/bin/bash

# 检查是否以 root 用户执行
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

# 提供选择菜单
echo "Jetson Nano の電源モードを選択してください："
echo "1. 全性能モード (10W)"
echo "2. 省電力モード (5W)"
read -p "選択 (1/2): " mode

# 根据用户选择执行操作
case $mode in
  1)
    echo "全性能モードに切り替えます..."
    sudo nvpmodel -m 0  # 10W モード
    sudo jetson_clocks   # クロックを最大に設定
    ;;
  2)
    echo "省電力モードに切り替えます..."
    sudo nvpmodel -m 1  # 5W モード
    sudo jetson_clocks --restore  # クロック設定をリセット
    ;;
  *)
    echo "無効な選択です。スクリプトを終了します。"
    exit 1
    ;;
esac

echo "設定が完了しました。現在の電源モード："
sudo nvpmodel -q  # 現在のモードを表示
