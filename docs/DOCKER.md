# Docker 部署

用 Docker 运行和 `scripts/serve.sh` 完全相同的服务配置，主机重启后自动启动。

## 前提

- 主机已装 Intel Gaudi 驱动和固件 **1.24.1**（[安装说明](INSTALL.md) 第 1 节），`hl-smi` 能看到卡。镜像里的 Gaudi 用户态库和 PyTorch 来自 Intel 的 1.24.1 官方镜像，必须与驱动版本一致。
- Docker 24 以上，带 compose 插件，`systemctl is-enabled docker` 为 `enabled`（开机自启动依赖它）。
- 不需要 Intel 的 habana container runtime：compose 直接把 `/dev/accel/accel0`、`/dev/accel/accel_controlD0` 和卡上网口的 RDMA 设备 `/dev/infiniband/uverbs0` 交给容器。少了最后一个，获取设备时会失败（日志里是 `Device acquire failed`，Habana 日志 `hcl.log` 里是 `failed to find matching Habana IB device (hbl_0)`）：即使单卡，获取设备也会初始化 HCCL。设备编号可用 `ls /dev/accel`、`ls -l /dev/infiniband/by-ibdev` 查看。
- 主机内存约 32 GiB 以上；镜像构建需要约 20 GB 磁盘（基础镜像约 6.5 GB，构建好的镜像约 10 GB）；设备编译缓存另占约 4 GB（在 `DATA_DIR`）。
- 同一张卡同一时间只能有一个服务。改用容器前先停掉 `scripts/serve.sh` 启动的服务（`scripts/stop.sh`）。

## 构建镜像

```bash
cp docker/docker.env.example docker/docker.env    # 填路径，见下一节
docker compose -f docker/compose.yaml --env-file docker/docker.env build
```

构建过程（基础镜像里没有 TPC 编译器，会从 Intel 的 apt 源装 `habanalabs-tools`）：拉取 `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` 基础镜像，安装 [requirements/lock-sdk-1.24.1.txt](../requirements/lock-sdk-1.24.1.txt) 中锁定版本的依赖，以可编辑模式安装 `vllm-core` 和 `vllm-gaudi`（即 `scripts/install.sh`），再编译三个本地库（`scripts/build_native.sh`）。下载基础镜像之后约 5 分钟。构建最后会做一次导入检查，依赖缺失时直接在构建阶段报错。国内网络可在 `docker.env` 里把 `PIP_INDEX_URL` 和 `APT_MIRROR` 换成就近的镜像源。

## 配置

两个文件：

| 文件 | 内容 |
|---|---|
| `config/serve.env` | 服务配置，与非容器部署是同一个文件（[配置说明](CONFIG.md)）。模型路径、`DATA_DIR`、端口、`KV_CACHE_GIB` 等都在这里改 |
| `docker/docker.env` | 只给 compose 用：`SERVE_ENV`（上面那个文件的绝对路径）、`MODEL_ROOT`（同时包含目标模型和草稿模型的目录）、`DATA_DIR`（必须与 `serve.env` 里的相同）、设备节点、镜像名 |

`MODEL_ROOT` 和 `DATA_DIR` 在容器里挂载到与主机相同的路径，所以 `serve.env` 不用为容器单独改。`serve.env` 里的 `HOST` 一般设 `0.0.0.0`；容器使用主机网络，端口就是 `serve.env` 里的 `PORT`。

## 启动、停止、查看

```bash
docker compose -f docker/compose.yaml --env-file docker/docker.env up -d      # 启动（之后开机自动启动）
docker logs -f gaudi2-vllm                                                    # 日志（另有一份在 $DATA_DIR/logs/serve-*.log）
scripts/smoke_test.sh --wait                                                  # 等预热完成并做一次请求检查
docker compose -f docker/compose.yaml --env-file docker/docker.env stop       # 停止（不再开机自启，直到下次 up/start）
docker compose -f docker/compose.yaml --env-file docker/docker.env up -d      # 改了 serve.env 后重新创建容器
```

- 启动到可用约 35–45 分钟：加载模型约 4 分钟，预热编译所有形状分桶约 30 分钟。设备端编译结果缓存在 `DATA_DIR/recipe-cache-*`，Dynamo 的动态维度记录在 `DATA_DIR/torch-cache`，保留 `DATA_DIR` 可以缩短之后的启动。镜像里的本地库是重新编译的，缓存目录名带库文件的哈希，所以从非容器部署换到容器后第一次启动会重新编译设备端 recipe。
- 预热完成、`/health` 正常后，容器会自动发几组单路和两路请求（`tools/prime.py`，结果在 `DATA_DIR/logs/serve-*-prime.log`），让预热没覆盖到的形状在这时编译。`serve.env` 里设 `PRIME_AFTER_START=0` 可以关闭。
- `restart: unless-stopped`：主机重启、Docker 重启或服务异常退出后都会自动重新启动；手动 `stop` 之后不会。模型在按需挂载的网络盘上时，容器会等模型文件出现（默认最多 30 分钟，`serve.env` 里的 `MODEL_WAIT_SECONDS`）再启动服务。
- `docker stop` 先发 TERM，150 秒后仍未退出才 KILL（引擎退出有时要接近 2 分钟）。

## 更新代码

仓库里的源码在构建时复制进镜像，更新代码后重新构建并重建容器：

```bash
git pull
docker compose -f docker/compose.yaml --env-file docker/docker.env build
docker compose -f docker/compose.yaml --env-file docker/docker.env up -d
```

只改了 Python 代码时本地库不变，recipe 缓存仍然有效；改了 `native/` 或 `vllm-gaudi/csrc` 下的内核会换一个新的缓存目录。
