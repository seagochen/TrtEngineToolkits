#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import paramiko
import os
import sys
import stat
from getpass import getpass
from pathlib import Path

# ==============================
# 配置部分
# ==============================

# 远程设备的SSH信息
REMOTE_USER = "ubuntu"  # 远程设备的用户名
REMOTE_HOST = "192.168.1.45"  # 远程设备的IP地址或主机名
REMOTE_DIR = "/home/ubuntu/Projects/Serverlite2"  # 远程设备上的目标目录

# 本地项目目录
LOCAL_PROJECT_DIR = "/home/orlando/Projects/ServerLite"  # 本地项目的根目录

# 清单文件路径
LIST_FILE = "list.txt"  # 清单文件名称（确保与脚本在同一目录，或提供绝对路径）

# SSH选项（如果需要使用特定的私钥，可以在这里指定）
SSH_KEY_PATH = None  # 例如: "/path/to/private/key" 或 None

# ==============================
# 脚本逻辑部分
# ==============================

def get_password():
    return getpass(f"请输入 {REMOTE_USER}@{REMOTE_HOST} 的SSH密码: ")

def connect_ssh(username, hostname, password=None, key_filename=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if key_filename:
            client.connect(hostname, username=username, key_filename=key_filename)
        else:
            client.connect(hostname, username=username, password=password)
        return client
    except Exception as e:
        print(f"无法连接到 {hostname}: {e}")
        sys.exit(1)

def sftp_delete_remote_item(sftp, remote_path):
    try:
        file_attr = sftp.stat(remote_path)
        if stat.S_ISDIR(file_attr.st_mode):
            # 如果是目录，递归删除其内容
            for entry in sftp.listdir_attr(remote_path):
                entry_path = os.path.join(remote_path, entry.filename).replace("\\", "/")
                sftp_delete_remote_item(sftp, entry_path)
            # 删除空目录
            try:
                sftp.rmdir(remote_path)
                print(f"删除远程目录: {remote_path}")
            except Exception as e:
                print(f"无法删除远程目录 {remote_path}: {e}")
        else:
            # 如果是文件，删除文件
            try:
                sftp.remove(remote_path)
                print(f"删除远程文件: {remote_path}")
            except Exception as e:
                print(f"无法删除远程文件 {remote_path}: {e}")
    except FileNotFoundError:
        print(f"远程路径 {remote_path} 不存在，跳过删除。")
    except Exception as e:
        print(f"删除远程路径 {remote_path} 失败: {e}")

def sftp_mkdirs(sftp, remote_path):
    dirs = []
    while len(remote_path) > 1:
        dirs.append(remote_path)
        remote_path, _ = os.path.split(remote_path)
    dirs = dirs[::-1]
    for directory in dirs:
        try:
            sftp.stat(directory)
        except IOError:
            try:
                sftp.mkdir(directory)
                print(f"创建远程目录: {directory}")
            except Exception as e:
                print(f"无法创建目录 {directory}: {e}")
                sys.exit(1)

def sftp_put_dir(sftp, local_dir, remote_dir):
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        rel_path = "" if rel_path == "." else rel_path
        remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
        sftp_mkdirs(sftp, remote_path)
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace("\\", "/")
            try:
                sftp.put(local_file, remote_file)
                print(f"上传文件: {local_file} -> {remote_file}")
            except Exception as e:
                print(f"无法上传文件 {local_file}: {e}")

def sftp_put_file(sftp, local_file, remote_file):
    remote_dir = os.path.dirname(remote_file)
    sftp_mkdirs(sftp, remote_dir)
    try:
        sftp.put(local_file, remote_file)
        print(f"上传文件: {local_file} -> {remote_file}")
    except Exception as e:
        print(f"无法上传文件 {local_file}: {e}")

def main():
    # 检查清单文件是否存在
    if not os.path.isfile(LIST_FILE):
        print(f"清单文件 {LIST_FILE} 不存在，请检查路径。")
        sys.exit(1)

    # 获取SSH密码
    password = None
    if not SSH_KEY_PATH:
        password = get_password()

    # 建立SSH连接
    ssh_client = connect_ssh(REMOTE_USER, REMOTE_HOST, password=password, key_filename=SSH_KEY_PATH)
    try:
        sftp = ssh_client.open_sftp()
    except Exception as e:
        print(f"无法打开SFTP连接: {e}")
        ssh_client.close()
        sys.exit(1)

    # 读取清单文件
    with open(LIST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 删除远程设备上 list.txt 中列出的内容
    print(f"正在删除远程设备上 {LIST_FILE} 中列出的内容...")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # 路径处理，确保目录以 '/' 结尾表示文件夹
        remote_path = os.path.join(REMOTE_DIR, line).replace("\\", "/")
        print(f"尝试删除远程路径: {remote_path}")
        sftp_delete_remote_item(sftp, remote_path)

    # 读取清单文件并同步文件/文件夹
    print(f"开始同步本地内容到远程设备...")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        local_path = os.path.join(LOCAL_PROJECT_DIR, line)
        remote_path = os.path.join(REMOTE_DIR, line).replace("\\", "/")

        if line.endswith("/"):
            # 处理文件夹拷贝
            if not os.path.isdir(local_path):
                print(f"本地文件夹 {local_path} 不存在，跳过。")
                continue
            print(f"同步文件夹 {line} 到远程设备...")
            sftp_put_dir(sftp, local_path, remote_path)
        else:
            # 处理单个文件拷贝
            if not os.path.isfile(local_path):
                print(f"本地文件 {local_path} 不存在，跳过。")
                continue
            print(f"同步文件 {line} 到远程设备...")
            sftp_put_file(sftp, local_path, remote_path)

    # 传输 list.txt 文件本身（可选，如果需要）
    # 如果不希望删除 list.txt 本身，可以将其从删除列表中排除，或者在同步后重新上传。
    # 以下示例假设需要上传 list.txt
    print(f"同步清单文件 {LIST_FILE} 到远程设备...")
    list_file_local = os.path.abspath(LIST_FILE)
    list_file_remote = os.path.join(REMOTE_DIR, LIST_FILE).replace("\\", "/")
    if os.path.isfile(list_file_local):
        sftp_put_file(sftp, list_file_local, list_file_remote)
    else:
        print(f"清单文件 {LIST_FILE} 本身不存在，跳过上传。")

    # 关闭连接
    sftp.close()
    ssh_client.close()

    print("代码同步完成！")

if __name__ == "__main__":
    main()
