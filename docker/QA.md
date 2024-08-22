
- [如何在Ubuntu环境中不使用sudo权限](#如何在ubuntu环境中不使用sudo权限)
  - [注意事项](#注意事项)


# 如何在Ubuntu环境中不使用sudo权限

可以通过将当前用户添加到 Docker 组来实现普通用户执行 Docker 命令，而无需使用 `sudo`。以下是步骤：

1. **创建 `docker` 组（如果尚未存在）**：
   ```bash
   sudo groupadd docker
   ```

2. **将当前用户添加到 `docker` 组**：
   ```bash
   sudo usermod -aG docker $USER
   ```

3. **重新启动你的终端会话**：为了使组更改生效，您需要重新登录，或者可以使用以下命令：
   ```bash
   newgrp docker
   ```

4. **验证是否生效**：可以运行以下命令来验证是否可以在不使用 `sudo` 的情况下运行 Docker 命令：
   ```bash
   docker run hello-world
   ```

   如果执行成功且无需 `sudo`，说明配置正确。

## 注意事项

- **安全性**：将用户添加到 Docker 组会赋予用户对 Docker 守护进程的完全访问权限，这意味着用户可以获得类似于 `root` 的权限。因此，请确保只将可信用户添加到 Docker 组中。
- **系统重启**：在某些情况下，您可能需要重启系统才能使这些更改完全生效。

执行以上步骤后，你应该能够通过普通用户而不是 `sudo` 来运行 Docker 命令。