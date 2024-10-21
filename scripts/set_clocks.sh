#!/bin/bash

# Check if the script is running as root
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

# If /root/.jetsonclocks_conf.txt is not present, create it
if [ ! -f /root/.jetsonclocks_conf.txt ]; then
    echo "Creating /root/.jetsonclocks_conf.txt..."
    sudo  /usr/bin/jetson_clocks --store

    if [ $? -ne 0 ]; then
        echo "エラーが発生しました。"
        exit 1
    else
        echo "設定ファイルを作成しました。"
    fi
fi

# Ask the user if they want to set the clocks
echo "Jetson Nano のクロックを設定してください："
echo "1. 自動設定"
echo "2. 省電力モード"
echo "3. 全性能モード"
echo "0. 現在のクロックを確認"
read -p "選択 (1/2/3): " mode

# Set the clocks based on the user's choice
case $mode in
  1)
    # Get the current power mode
    MODE=$(sudo nvpmodel -q | grep -oP '(?<=Power Mode: ).*')

    # Set the clocks based on the current power mode
    if [[ "$MODE" == "MAXN" ]]; then
        echo "全性能モードです。クロックを最大に設定します..."
        sudo jetson_clocks  # 最大频率
    elif [[ "$MODE" == "MODE_10W" ]]; then
        echo "省電力モードです。クロックをリセットします..."
        sudo jetson_clocks --restore  # 恢复默认频率
    else
        echo "不明なモードです。手動で確認してください。"
        exit 1
    fi

    echo "クロックを設定しました。"
    ;;
  2)
    echo "省電力モードに設定します..."
    sudo jetson_clocks --store  # 保存当前频率
    echo "省電力モードに設定しました。"
    ;;
  3)
    echo "全性能モードに設定します..."
    sudo jetson_clocks  # 最大频率
    echo "全性能モードに設定しました。"
    ;;
  0)
    echo "現在のクロックを確認します..."
    sudo jetson_clocks --show
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