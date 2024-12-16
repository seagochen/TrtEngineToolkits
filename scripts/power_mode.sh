#!/bin/bash

# スクリプトを root ユーザーとして実行しているか確認
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

# nvpmodel がインストールされているか確認
if ! command -v nvpmodel &> /dev/null; then
  echo "エラー: nvpmodel がインストールされていません。"
  echo "インストールするには以下のコマンドを実行してください:"
  echo "sudo apt-get install nvpmodel"
  exit 1
fi

# 現在の電源モードを表示
current_mode=$(nvpmodel -q | grep "Mode" | awk '{print $2}')
echo "現在の電源モード: $current_mode"

# 選択メニューを表示
echo "Jetson Nano の電源モードを選択してください："
echo "1. 全性能モード (MAXN)"
echo "2. 省電力モード (10W)"
read -p "選択 (1/2): " mode

case $mode in
  1)
    echo "全性能モードに切り替えます..."
    nvpmodel -m 0  # MAXN モードに切り替え
    ;;
  2)
    echo "省電力モードに切り替えます..."
    nvpmodel -m 1  # 10W モードに切り替え
    ;;
  *)
    echo "無効な選択です。スクリプトを終了します。"
    exit 1
    ;;
esac

# 変更を有効にするためにシステムを再起動が必要
echo "変更を有効にするためにシステムを再起動する必要があります。"
while true; do
  read -p "再起動しますか？ (y/n): " reboot
  case "$reboot" in
    [Yy]* )
      reboot
      ;;
    [Nn]* )
      echo "再起動をキャンセルしました。"
      exit 0
      ;;
    * )
      echo "有効な入力をしてください (y/n)。"
      ;;
  esac
done
