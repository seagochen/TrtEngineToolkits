#!/bin/bash

# スクリプトを root 権限で実行しているか確認
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

CONFIG_FILE="/root/.jetsonclocks_conf.txt"

# 設定ファイルの存在を確認し、なければ作成
create_config() {
    echo "Creating $CONFIG_FILE..."
    jetson_clocks --store

    if [ $? -ne 0 ]; then
        echo "エラーが発生しました。"
        exit 1
    else
        echo "設定ファイルを作成しました。"
    fi
}

if [ ! -f "$CONFIG_FILE" ]; then
    create_config
fi

# クラック設定の関数
set_clocks() {
    local mode=$1
    case $mode in
        1)
            # 自動設定
            MODE=$(nvpmodel -q | grep -oP '(?<=Power Mode: ).*')

            if [[ "$MODE" == "MAXN" ]]; then
                echo "全性能モードです。クロックを最大に設定します..."
                jetson_clocks
            elif [[ "$MODE" == "MODE_10W" ]]; then
                echo "省電力モードです。クロックをリセットします..."
                jetson_clocks --restore
            else
                echo "不明なモードです。手動で確認してください。"
                exit 1
            fi

            echo "クロックを設定しました。"
            ;;
        2)
            # 省電力モード
            echo "省電力モードに設定します..."
            jetson_clocks --store
            echo "省電力モードに設定しました。"
            ;;
        3)
            # 全性能モード
            echo "全性能モードに設定します..."
            jetson_clocks
            echo "全性能モードに設定しました。"
            ;;
        0)
            # 現在のクロックを確認
            echo "現在のクロックを確認します..."
            jetson_clocks --show
            ;;
        *)
            echo "無効な選択です。スクリプトを終了します。"
            exit 1
            ;;
    esac
}

# ファン制御の関数
control_fan() {
    local action=$1
    case $action in
        on)
            echo "ファンを強制起動します..."
            jetson_clocks --fan
            if [ $? -eq 0 ]; then
                echo "ファンを強制起動しました。"
            else
                echo "ファンの起動に失敗しました。"
            fi
            ;;
        off)
            echo "ファンを停止します..."
            jetson_clocks --restore
            if [ $? -eq 0 ]; then
                echo "ファンを停止しました。"
            else
                echo "ファンの停止に失敗しました。"
            fi
            ;;
        *)
            echo "無効なファン制御オプションです。"
            ;;
    esac
}

# メインメニューの表示
show_menu() {
    echo "Jetson Nano のクロックおよびファン設定メニュー："
    echo "1. 自動設定"
    echo "2. 省電力モード"
    echo "3. 全性能モード"
    echo "4. ファンを強制起動"
    echo "5. ファンを停止"
    echo "0. 現在のクロックを確認"
}

# ユーザー入力の取得と処理
while true; do
    show_menu
    read -p "選択 (0-5): " choice

    case $choice in
        1|2|3|0)
            set_clocks "$choice"
            ;;
        4)
            control_fan "on"
            ;;
        5)
            control_fan "off"
            ;;
        6)
            # 追加のオプションが必要な場合に備えて
            echo "このオプションは未実装です。"
            ;;
        *)
            echo "無効な選択です。再度選択してください。"
            continue
            ;;
    esac

    # 再起動の確認
    echo "変更を有効にするためにシステムを再起動してください。"
    read -p "再起動しますか？ (y/n): " reboot_choice

    case "$reboot_choice" in
        y|Y)
            echo "システムを再起動します..."
            reboot
            ;;
        n|N)
            echo "再起動をキャンセルしました。"
            ;;
        *)
            echo "無効な入力です。再起動をキャンセルしました。"
            ;;
    esac

    # ループを終了
    break
done
