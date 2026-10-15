# 源码快照说明

本次快照于 2026-10-14 从目标服务器只读导出。Gaudi 插件和 vLLM 主程序均包含已跟踪源码及通过过滤的未跟踪源码，不使用旧的本地离线候选替代服务器现有文件。

`vllm-core/vllm/` 以服务器实际安装的 Python 文件覆盖 checkout 中同名文件。覆盖差异列在 `SOURCE_SNAPSHOT.json` 的 `components` 中；其余构建配置、C++ 源码及测试来自对应源码 checkout。不是复制完整 site-packages 或包含依赖包的运行环境。

导出时按每个文件的大小/mtime检查并发修改，记录最终字节的 SHA-256；哈希清单可用于检查仓库是否仍与本次导出一致。它不是数字签名，也不构成整个运行环境的可重复构建证明。

本次未导出服务器 Git 历史，而是记录来源 commit 并创建独立快照提交；原有版权/许可文件保留。CI workflow、原始实验 evidence、备份、日志、二进制、私有环境配置和模型权重不在源码清单中。

后续更新应重新读取现用源码、检查运行包差异、扫描敏感信息并更新清单。不要直接把本地候选、模型文件或带有凭据的部署脚本加入公开仓库。

## 2026-10-15 更新

`vllm-gaudi/vllm_gaudi/` 下 11 个文件按服务器现用版本更新（6 个修改、5 个新增，均与服务器 `/opt` 中的字节逐一核对过 SHA-256），对应 [优化说明](OPTIMIZATION.md) 第 9 节的 v6–v10 改动和此前的草稿输入打包。清单中这些文件的哈希与 `captured_at_utc` 已同步更新。

`native/tpc-kernel-shim/` 是新增源码，不在快照清单中：部署版本的默认库列表写的是本机路径，仓库版本改为从 `KIRO_TPC_LIBS` 读取（未设置时只加载官方库），其余逻辑相同。

## 2026-10-15 更新（v11）

`vllm-gaudi/vllm_gaudi/` 下 5 个文件按服务器现用 v11 版本更新（4 个修改、新增 `v1/worker/kiro_v11.py`），均与服务器 `/opt` 中的字节核对过 SHA-256，对应 [优化说明](OPTIMIZATION.md) 第 11 节。清单中的哈希与 `captured_at_utc` 已同步。

## 2026-10-15 更新（v11b–v13）

`vllm-gaudi/vllm_gaudi/` 下 5 个文件（`envs.py`、`models/qwen3_next.py`、`v1/worker/hpu_model_runner.py`、`kiro_v6.py`、`kiro_v11.py`）和 `vllm-gaudi/flashinfer_gaudi/` 下 3 个文件（`_native.py`、`_config.py`、`gdn_decode.py`）按服务器现用 v13 版本更新，均与服务器 `/opt` 中的字节核对过 SHA-256，对应 [优化说明](OPTIMIZATION.md) 第 12–14 节。清单中的哈希与 `captured_at_utc` 已同步。

`native/gdn-fused-kernel/` 是新增源码，不在快照清单中。`vllm-gaudi/csrc/flashinfer_gaudi/` 保持服务器原样；部署的 v12 内核库是在它的副本上运行 `native/gdn-fused-kernel/flashinfer_gaudi/patch_fused_src.py` 后编译的，打补丁后的源码树与部署时使用的逐字节一致。

## 2026-10-15 更新（v14）

`vllm-gaudi/flashinfer_gaudi/_config.py` 与 `gdn_decode.py` 按服务器现用 v14 版本更新（融合 GDN 核心先取初始状态，[优化说明](OPTIMIZATION.md) 第 15 节），已与服务器 `/opt` 中的字节核对 SHA-256，清单同步。

## 2026-10-15 整理为可直接使用的工程

- `vllm-gaudi/csrc/flashinfer_gaudi/` 直接包含融合 GDN kernel 和 8 行 prepared 核心（即 `native/gdn-fused-kernel/flashinfer_gaudi/patch_fused_src.py` 的结果），正常编译即得到部署所用的内核库。这 8 个文件从此与服务器 `/opt` 里未打补丁的源码不同，与部署 v12 库时使用的源码逐字节一致。
- 新增 `scripts/`、`config/`、`requirements/` 与 `docs/INSTALL.md`、`docs/CONFIG.md`；`tools/bench/` 的服务地址、模型名和 API key 改为从环境变量读取。
- 清单中 `vllm-gaudi/` 下变化或新增文件的哈希已同步。
- 验证：在实测机器上按 `docs/INSTALL.md` 的步骤，在一个新目录里安装、编译并用 `scripts/serve.sh` 启动（首次预热 1761 秒）。冒烟测试通过，每轮耗时与 v14 相同：单路 26.22 ms，两路 29.94 ms，3 万 token 30.16 ms。之后生产服务直接运行这份安装。

## 2026-10-15 更新（v15）

新增 `vllm-gaudi/vllm_gaudi/extension/dense_decode_attn.py`，修改 `envs.py`、`extension/ops.py`、`attention/ops/hpu_paged_attn.py` 和 `scripts/serve.sh`（整段计算的共享前缀注意力，[优化说明](OPTIMIZATION.md) 第 17 节）。均与生产服务所用安装（本仓库的一份副本）逐字节核对过，清单已同步。
