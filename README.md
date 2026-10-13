# Gaudi2 Single-Card vLLM

面向 **单张 Intel Gaudi2（96 GiB HBM）** 的 vLLM 源码快照与本地适配工程，重点关注 Qwen3.5 系列的 FP8 推理、非 greedy 投机解码，以及主机到 HPU 的执行开销。

本仓库保存 **2026-10-14 从服务器读取的实际源码**，包含未提交的本地改动。它不是 Intel 或 vLLM 官方发行版，也不是“一键安装、所有功能均已验证”的稳定发行包。

## 项目来源

本项目基于 1Cat-vLLM-Gaudi 工程分支、[vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) 硬件插件及 [vllm-project/vllm](https://github.com/vllm-project/vllm) 推理引擎。保留各组件原有 LICENSE、版权声明、文档及第三方许可；新增的单卡适配与采样扩展按所在文件的声明处理。

这是一次源码发布，不是向上游提交的 PR。部分本地适配和文档整理使用了 AI 辅助，仍需要人工审查与目标环境验收。

## 仓库结构

```text
vllm-core/           配套 vLLM 主程序源码，叠加服务器实际安装的 Python 文件
vllm-gaudi/          服务器上的 Gaudi 插件、FlashInfer-Gaudi、C++/TPC 核及相关测试
native/mtp-sampling/ 本地随机拒绝纠正/输出打包扩展源码
docs/               本次快照说明与使用边界
SOURCE_SNAPSHOT.json 源码清单、SHA-256 与来源版本
```

Gaudi 插件保留原 `README.md`，详细模型适配说明见 [组件介绍](vllm-gaudi/README.md)。主程序原文档见 [vLLM 文档源码](vllm-core/docs/)。

## 本地适配的主要方向

- **Qwen3.5 / Qwen3-Next 混合架构**：完整注意力、Gated DeltaNet、卷积历史及投机验证检查点的 HPU 路径。
- **FP8 推理**：Gaudi2 权重范围/布局处理、动态量化与 FP8 GEMM。模型 FP8 线性不等于所有算子或 KV 缓存都使用 FP8。
- **DFlash2 与 MTP**：包含本地草稿生成、目标验证、状态回滚和非 greedy 随机拒绝纠正实现。不同模型、投机宽度与 batch 的资格并不相同，不能任意互换。
- **CPU 提交链路**：decode 输入准备、注意力组路由、共享前缀准备、静态 guard 筛选及部分环境变量缓存。较新的 fast-decode 路径仍需完整验收。
- **OpenAI 兼容服务**：沿用 vLLM API；自动工具调用需启用相应选项并选择与模型模板匹配的 parser，本地配置使用过 `qwen3_xml`。
- **诊断入口**：保留有界计时和 profiling 支持。诊断配置不是性能推荐配置，不应在正式测速时默认开启全量 trace。

上述列表描述源码中存在的实现，不代表每个组合均经过完整模型、长上下文、并发、工具调用或取消/抢占验收。

## 快照对应的软件栈

| 组件 | 服务器对应版本 / 基线 |
|---|---|
| 硬件 | Intel Gaudi2，单卡，TP=1 / PP=1 |
| Python | 3.12 系列 |
| Gaudi 驱动 | `1.24.1-b336d5e` |
| Habana Torch plugin | `1.24.1.482` |
| PyTorch | `2.11.0a0+git009b5f6` |
| vLLM | `0.28.1rc1.dev100+gfe755c889.d20261009.empty` |
| vLLM 核心 Git 基线 | `fe755c88995ad468882517b6c4bdd60138d46a3a` |
| Gaudi 插件 Git 基线 | `57a50be3f7fc3a61c0a89c9ae3a3612072269b99` |

Git 基线不包含全部本地改动；请以本仓库文件及 `SOURCE_SNAPSHOT.json` 为准。主程序目录叠加了服务器安装包中存在差异的 Python 文件，避免仅导出旧 checkout。

该环境使用 **non-lazy / eager 加区域编译**。`torch.compile`、Native Replay 和 HPUGraph 是不同机制，不能统称“已开启 Graph”。PyTorch / Habana 插件 / Synapse 与自定义 native 库的版本和 ABI 必须匹配。

## 使用前须知

1. 准备匹配的 Gaudi 软件栈及 SDK。不要为安装本项目随意刷固件、替换驱动，或让通用 CUDA 安装命令覆盖 HPU 版 PyTorch。
2. 分别阅读 [vLLM 核心说明](vllm-core/README.md) 和 [Gaudi 组件构建配置](vllm-gaudi/pyproject.toml)。这是多组件快照，仓库根目录不能直接作为单一 Python 包安装。
3. 自定义 FlashInfer-Gaudi/TPC 算子及 [随机采样扩展](native/mtp-sampling/) 需要按目标环境构建。仓库不包含服务器编译出的 `.so`、wheel 或 SDK 二进制。
4. 设置本机的模型目录、动态库路径、kernel 搜索路径及日志目录。源码中保留的部分默认路径来自原环境，不是可移植的部署承诺。
5. 先做组件数值验证，再做非 greedy 的短请求、工具调用和流式验收，最后比较同模型、同参数的热请求基线。不要直接开启所有实验 flag 或跳过预热后报告峰值。

请参考 [Intel Gaudi 文档](https://docs.habana.ai/en/latest/index.html) 与 [vLLM Gaudi 文档](https://docs.vllm.ai/projects/gaudi/en/0.26.0/index.html)；后者是单独的文档版本，不等于本快照的源码基线。

## 当前验证边界

服务器最新启动脚本指向 v5 **诊断 / smoke** 配置，含跳过预热、profiling 与 fast-decode 检查选项；本次只读取源码，不启动、停止或改动服务器服务。该配置没有被包装为默认生产启动器。

最近独立实验已验证真实前 8 层混合组的预录对照，以及完整 48 组 GDN 状态池的输出/缓存一致性；但 **预录没有显示端到端净提速**。对应未部署候选、私有重编框架和本地实验结果不属于这份服务器快照，未混入发布目录。

本次上传本身不构成新的模型测速或全量测试。没有宣称达到硬件性能上限，没有把低 AIP-Util 直接归因为 CPU 等待，也没有承诺投机解码必然优于普通解码。

## 发布内容与排除项

包含公开源代码、原始测试/构建入口，以及服务器新增的源文件。排除模型权重、缓存、编译产物、运行日志、trace、备份副本、机器凭据、私有环境文件和原仓库的 CI workflow，避免导入时自动触发不相关构建。

源码校验方法：按 `SOURCE_SNAPSHOT.json` 中的相对路径读取文件并计算 SHA-256。仓库根部新写的介绍文件不属于原服务器源码，因此不列入服务器源文件清单。

## 许可

主体来源项目采用 Apache-2.0；分别见 [vLLM LICENSE](vllm-core/LICENSE) 与 [Gaudi 插件 LICENSE](vllm-gaudi/LICENSE)。第三方目录和文件保留各自许可，不将其统一重授为其他许可。
