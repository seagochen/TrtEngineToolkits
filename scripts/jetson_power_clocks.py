#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

CONFIG_FILE = "/root/.jetsonclocks_conf.txt"
BACKUP_FILE = "/root/.jetsonclocks_conf.backup.txt"

# ---------- utils ----------
def run(cmd, check=True, capture=False, text=True):
    return subprocess.run(
        cmd, check=check, capture_output=capture, text=text
    )

def cmd_exists(name: str) -> bool:
    return run(["bash", "-lc", f"command -v {name}"], check=False).returncode == 0

def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    comp = Path("/proc/device-tree/compatible")
    if comp.exists() and "tegra" in comp.read_text(errors="ignore").lower():
        return True
    return False

def require_root():
    if os.geteuid() != 0:
        print("このスクリプトを実行するには root 権限が必要です。")
        raise SystemExit(1)

# ---------- nvpmodel ----------
def ensure_nvpmodel():
    if not cmd_exists("nvpmodel"):
        print("エラー: nvpmodel がインストールされていません。")
        print("インストールするには以下のコマンドを実行してください:")
        print("  sudo apt-get update && sudo apt-get install -y nvpmodel")
        raise SystemExit(1)

def get_nvpmodel_mode() -> str:
    # 例: "Mode: 0" を抽出
    p = run(["nvpmodel", "-q"], check=False, capture=True)
    out = p.stdout or ""
    for line in out.splitlines():
        if "Mode" in line:
            # "Mode: 0" → 0
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("Mode"):
                return parts[-1]
    return "不明"

def set_nvpmodel(mode_idx: int):
    print(f"nvpmodel モードを {mode_idx} に切り替えます…")
    run(["nvpmodel", "-m", str(mode_idx)])

# ---------- jetson_clocks ----------
def ensure_jetson_clocks():
    if not cmd_exists("jetson_clocks"):
        print("エラー: jetson_clocks が見つかりません。")
        print("JetPack 付属のユーティリティです。SDK Manager で Jetson の開発者ツールを有効化してください。")
        raise SystemExit(1)

def setup_jetson_clocks_config():
    print("クロック設定ファイルの確認とセットアップ中...")
    if not Path(CONFIG_FILE).exists():
        print(f"クロック設定ファイル ({CONFIG_FILE}) が存在しません。新しく作成します…")
        if run(["jetson_clocks", "--store"], check=False).returncode == 0:
            print(f"設定ファイルを作成しました: {CONFIG_FILE}")
        else:
            print("エラー: 設定ファイルの作成に失敗しました。'jetson_clocks --store' を手動で実行してみてください。")

    # 初回バックアップ
    if Path(CONFIG_FILE).exists() and not Path(BACKUP_FILE).exists():
        print(f"初回バックアップ ({BACKUP_FILE}) を作成中…")
        cp = run(["cp", CONFIG_FILE, BACKUP_FILE], check=False)
        if cp.returncode == 0:
            run(["chmod", "444", BACKUP_FILE], check=False)
            print(f"バックアップを作成しました: {BACKUP_FILE}")
        else:
            print("警告: バックアップの作成に失敗しました。")

def jetson_clocks_max():
    print("パフォーマンスを最大化し、ファンを強制起動します…")
    if run(["jetson_clocks"], check=False).returncode == 0:
        print("パフォーマンスの最大化とファンの強制起動に成功しました。")
    else:
        print("エラー: パフォーマンスの最大化に失敗しました。")

def jetson_clocks_restore():
    print("パフォーマンスとファンを元の(保存された)状態に戻します…")
    if not Path(CONFIG_FILE).exists():
        print(f"エラー: 復元するための設定ファイル ({CONFIG_FILE}) が見つかりません。")
        print("先に「パフォーマンス最大化」を実行するか、手動で 'sudo jetson_clocks --store' を実行してください。")
        return
    if run(["jetson_clocks", "--restore"], check=False).returncode == 0:
        print("パフォーマンスとファンの元の状態への復元に成功しました。")
    else:
        print("エラー: 復元に失敗しました。")

def jetson_clocks_show():
    print("現在のクロック状態を表示します…")
    if run(["jetson_clocks", "--show"], check=False).returncode != 0:
        print("エラー: クロック状態の表示に失敗しました。")

# ---------- UI ----------
def ask_reboot():
    while True:
        ans = input("変更を有効にするためにシステムを再起動する必要があります。再起動しますか？ (y/n): ").strip().lower()
        if ans.startswith("y"):
            print("再起動します…")
            run(["reboot"], check=False)
            return
        if ans.startswith("n"):
            print("再起動をキャンセルしました。")
            return
        print("有効な入力をしてください (y/n)。")

def menu():
    print("")
    print("Jetson 電源/パフォーマンス & ファン・クロック制御")
    print("----------------------------------------------")
    print(" 電源/性能 (nvpmodel)")
    print("   1) 全性能モード (MAXN)        -> nvpmodel -m 0")
    print("   2) 省電力モード (10W)         -> nvpmodel -m 1")
    print("")
    print(" クロック/ファン (jetson_clocks)")
    print("   3) パフォーマンス最大化（クロック最大 & ファン最大）")
    print("   4) 通常パフォーマンスへ復元（保存済み状態へ）")
    print("   5) 現在のクロック状態を表示")
    print("")
    print(" 0) 終了")
    print("----------------------------------------------")

def main():
    require_root()

    if not is_jetson():
        print("このスクリプトは NVIDIA Jetson 専用です。終了します。")
        raise SystemExit(1)

    # 工具存在チェック
    # nvpmodel は電源モードに必要、jetson_clocks はクロック操作に必要
    nvp_ok = cmd_exists("nvpmodel")
    jc_ok = cmd_exists("jetson_clocks")

    if not nvp_ok:
        print("警告: 'nvpmodel' が見つかりません。電源モードの切替は使用できません。")
        print("  sudo apt-get update && sudo apt-get install -y nvpmodel")
    if not jc_ok:
        print("警告: 'jetson_clocks' が見つかりません。クロック/ファンの制御は使用できません。")
        print("  SDK Manager で Jetson の開発者ツールをインストールしてください。")

    # 事前セットアップ
    if jc_ok:
        setup_jetson_clocks_config()

    # 初回情報
    if nvp_ok:
        try:
            cur = get_nvpmodel_mode()
            print(f"現在の電源モード: {cur}")
        except Exception:
            print("現在の電源モード取得に失敗しました。")

    # 交互ループ
    while True:
        menu()
        choice = input("選択 (0-5): ").strip()

        if choice == "0":
            print("スクリプトを終了します。")
            break

        elif choice == "1":
            if not nvp_ok:
                print("nvpmodel が利用できません。")
                continue
            try:
                set_nvpmodel(0)  # MAXN
                ask_reboot()
            except subprocess.CalledProcessError:
                print("エラー: モード切替に失敗しました。")

        elif choice == "2":
            if not nvp_ok:
                print("nvpmodel が利用できません。")
                continue
            try:
                set_nvpmodel(1)  # 10W
                ask_reboot()
            except subprocess.CalledProcessError:
                print("エラー: モード切替に失敗しました。")

        elif choice == "3":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_max()

        elif choice == "4":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_restore()

        elif choice == "5":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_show()

        else:
            print("無効な選択です。0〜5 の数字を入力してください。")

if __name__ == "__main__":
    main()
