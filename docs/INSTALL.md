# 安装说明

本项目在下面这套环境上构建和实测：Ubuntu 24.04.4、Python 3.12.3、Intel Gaudi SDK 1.24.1（驱动 1.24.1-482）、Intel 为 1.24.1 发布的 PyTorch（`torch 2.11.0a0+git009b5f6`、`habana-torch-plugin 1.24.1.482`），单张 Gaudi2 HL-225。其他版本没有测试过：自定义 TPC 内核和 PyTorch 扩展都要与运行时版本完全匹配。

## 1. Gaudi 驱动、固件与工具

按 Intel 官方文档（<https://docs.habana.ai>，Installation Guide，选 1.24.1）安装 base 组件。本项目用到的 apt 包：

| 包 | 用途 |
|---|---|
| `habanalabs-dkms`、`habanalabs-firmware`、`habanalabs-firmware-tools` | 驱动和固件 |
| `habanalabs-thunk` | 用户态驱动库 |
| `habanalabs-graph` | 图编译器与 `/usr/include/habanalabs` 头文件 |
| `habanalabs-tools` | `hl-smi`、TPC 编译器 `tpc-clang` |

装好后 `hl-smi` 应能看到这张卡。

另外需要：

```bash
sudo apt install -y build-essential cmake libgoogle-perftools4t64 python3.12-venv
```

## 2. PyTorch（Intel Gaudi 版）

不能用 PyPI 上的 torch。下载 1.24.1 的 PyTorch 模块包，按包里的 `install.sh` 安装：

```bash
curl -LO https://vault.habana.ai/artifactory/gaudi-pt-modules/1.24.1/482/pytorch/ubuntu2404/pytorch_modules-v2.11.0_1.24.1_482.tgz
mkdir pt && tar xzf pytorch_modules-v2.11.0_1.24.1_482.tgz -C pt && cd pt
sudo ./install.sh 1.24.1 482      # 装到系统的 python3.12
```

它会安装 `torch`、`habana-torch-plugin`、`habana-torch-dataloader`、`habana-gpu-migration`、`habana-pyhlml` 等。装完后用 `scripts/check_env.sh` 检查版本。

## 3. 本项目的 Python 包

```bash
scripts/install.sh --venv .venv
```

- 新建一个 `--system-site-packages` 的 venv，从系统继承第 2 步装好的 Gaudi 版 PyTorch。也可以不加 `--venv`，直接装进系统的 `python3.12`（需要 `--break-system-packages`，脚本会自动加）。
- 按 [requirements/lock-sdk-1.24.1.txt](../requirements/lock-sdk-1.24.1.txt) 安装依赖，版本与实测环境一致；加了 `--no-deps`，不会改动 torch。
- `vllm-core` 以 `VLLM_TARGET_DEVICE=empty` 安装，不编译 CUDA 部分。`vllm-gaudi`（包含 `vllm_gaudi`、`flashinfer_gaudi` 和请求过滤中间件）以可编辑模式安装，运行的就是仓库里的源码，改了 Python 代码重启服务即可生效。
- 脚本最后会确认 `vllm`、`vllm_gaudi`、`flashinfer_gaudi` 都从本仓库导入。如果系统里另有一份已安装的 vLLM，它们会被仓库版本覆盖（venv 里优先），不会被删除。
- 所用的 Python 记录在 `.python`，后续脚本都会用它。

## 4. 本地库

```bash
scripts/build_native.sh        # MAX_JOBS=2 可以降低编译时的内存占用
```

| 产物 | 来源 |
|---|---|
| `build/native/libmtp_sampling_kernels_v2.so`、`build/native/mtp_sampling_v2.*.so` | `native/mtp-sampling`：随机拒绝采样的 TPC 内核和 PyTorch 算子 |
| `build/native/libflashinfer_gaudi_kernels.so`、`build/native/flashinfer_gaudi_ops.*.so` | `vllm-gaudi/csrc/flashinfer_gaudi`：GDN、DFlash2 等 TPC 内核（含融合 GDN 核心）和 PyTorch 算子 |
| `build/native/libkiro_tpc_shim.so` | `native/tpc-kernel-shim`：把三个内核库合成一个文件交给图编译器 |

编译只在主机上进行，不占用设备，服务运行时也可以编译。FlashInfer-Gaudi 的库故意不放进 `vllm-gaudi/flashinfer_gaudi/lib/`：放在那里时，包在导入时会把它加进 `GC_KERNEL_PATH`，而编译缓存要求 `GC_KERNEL_PATH` 只能是一个文件。最后的检查会核对三个库的全部内核都能通过转接层找到（实测 2 + 10 + 3250 = 3262）。

## 5. 模型

| 用途 | 实测使用 |
|---|---|
| 目标模型 | Qwen3.8-27B 的 FP8 量化（block 128×128，E4M3），保留 MTP 层权重（`model-mtp.safetensors`），约 27 GB |
| 草稿模型 | 对应 Qwen3.8-27B 的 DFlash2 草稿模型，约 3.4 GB |

把两个目录的路径填到 `config/serve.env` 的 `TARGET_MODEL` 和 `DRAFT_MODEL`。服务以离线模式运行（`HF_HUB_OFFLINE=1`），不会联网下载。

## 6. 第一次启动

```bash
cp config/serve.env.example config/serve.env   # 编辑模型路径
scripts/serve.sh --daemon
scripts/smoke_test.sh --wait
```

第一次启动要编译所有形状分桶，设备端 recipe 也全部是新编译的，总共约 30–45 分钟（实测：按本文步骤在干净目录安装后首次启动，加载约 3.5 分钟，预热 1761 秒），期间日志里会看到 `Prompt warmup processing` 和 `Decode warmup processing` 的进度。之后的启动会复用 `DATA_DIR` 里的 recipe 缓存和 Dynamo 的动态维度记录，大约 25 分钟。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `check_env.sh` 报 torch / habana-torch-plugin 版本不对 | 第 2 步没有装成 1.24.1 对应的包，或者 venv 没有继承系统包（要用 `--system-site-packages`） |
| 启动时报设备无法获取（acquire / busy），或长时间卡在初始化 | 卡被别的进程占用（Gaudi 同一时间只能被一个进程打开）。`hl-smi` 查看占用进程 |
| 启动时报 `GC_KERNEL_PATH: Path is not a regular file` | 没有用 `serve.sh`，自己写启动命令时在 `GC_KERNEL_PATH` 里用冒号拼了多个库。应指向 `build/native/libkiro_tpc_shim.so`，多个库写在 `KIRO_TPC_LIBS` |
| 以普通用户运行时在初始化或分配内存时失败 | Gaudi 需要锁定（pin）主机内存。用 root 运行（`sudo scripts/serve.sh ...`，此时在 `serve.env` 里显式设置 `DATA_DIR`），或把该用户的 memlock 上限设为 unlimited。实测部署以 root 运行 |
| 加载模型时进程被系统杀掉 | 主机内存不足。权重要经过主机内存，32 GiB 的机器上不要同时运行其他占内存的程序 |
| 预热比预期慢很多 | 第一次启动或内核库刚重新编译过，需要重新编译设备端 recipe，属于正常情况 |
| 请求返回 “HPU DFlash2 ... does not safely support ...” | 请求带了投机解码路径不支持的参数（见 README「调用」） |
| `install.sh` 报某个包从仓库以外的位置导入 | 不用 venv、直接装进系统时，旧的 vLLM 安装可能仍排在前面。改用 `--venv` |
