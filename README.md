<div align="center">

# Gaudi2 Single-Card vLLM

**在单张 Intel Gaudi2 上跑 27B 混合架构模型：FP8 推理、非 greedy 投机解码、262K 上下文**

![hardware](https://img.shields.io/badge/hardware-Intel%20Gaudi2%20%C3%97%201-0071c5)
![vllm](https://img.shields.io/badge/vLLM-0.28.1rc1-4b8bbe)
![synapse](https://img.shields.io/badge/SynapseAI-1.24.1-5c6bc0)
![license](https://img.shields.io/badge/license-Apache--2.0-2ea44f)
![status](https://img.shields.io/badge/status-source%20snapshot-orange)

[优化说明与实测](docs/OPTIMIZATION.md) · [离线验证](validation/README.md) · [测速与 trace 工具](tools/README.md) · [启动示例](examples/launcher-v5.example.sh) · [快照说明](docs/SNAPSHOT.md)

</div>

---

这是一份面向 **单张 Intel Gaudi2（96 GiB HBM）** 的 vLLM 源码快照与本地适配工程，重点是 Qwen3.5 架构模型的 FP8 推理、DFlash2 / MTP 非 greedy 投机解码，以及主机到 HPU 的执行开销。

> [!IMPORTANT]
> 本仓库保存的是从服务器读取的实际源码，包含未提交的本地改动。它不是 Intel 或 vLLM 的官方发行版，也不是一键安装、所有功能都验证过的稳定包。下面的数字都来自同一台机器，条件写在表格旁边；没有实测的地方会明确标注。

## 一眼看结果

测试条件：单卡，TP=1 / PP=1，本地命名的 Qwen3.8-27B-FP8（Qwen3.5 架构）加 DFlash2 草稿模型，非 greedy 采样，关闭 thinking。

| 场景 | 优化前 | 现在 | 变化 |
|---|---:|---:|---|
| 单路，86K 上下文 | 9.8 tok/s | **47.6 tok/s** | 约 5 倍 |
| 单路，30K 上下文 | 45.2 tok/s | **59–65 tok/s** | 每轮耗时 −22% |
| 2 路并发，短上下文，每轮耗时 | 约 64 ms | **约 52 ms** | −19% |
| 单路，短上下文，每轮耗时 | 43.5–44.8 ms | **42.3 ms** | −4% |
| 代码类提示词，2 路合计 | 141–173 tok/s | **约 209 tok/s** | |
| 最大上下文 | 65.5K 以上掉入慢路径 | **262K 全程走预热形状** | 256K 本身未实测 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/decode-round-dark.svg">
  <img alt="每轮解码耗时随上下文长度的变化" src="docs/assets/decode-round-light.svg">
</picture>

<sub>圆点是实测值。优化前在 30K 和 91K 之间没有测量点，连线只是示意。完整数据和测试条件见 <a href="docs/OPTIMIZATION.md">优化说明</a>。</sub>

**长上下文是这一轮的主要收益，短上下文单路已经接近硬件上限。** 单路一轮约 42 ms，其中设备实际在算 28 ms，而这 28 ms 里有 15 ms 是把全部权重读一遍，已经顶到 HBM 带宽。

## 这一轮做了什么

| 改动 | 开关 | 实测效果 |
|---|---|---|
| **验证 token 共用上下文 KV**：一轮验证的 8 个 token 原来各自 gather 一遍上下文，改为每个请求只读一次 | `VLLM_HPU_SHARED_PREFIX_DECODE=1` | 30K 时设备侧 gather 从 10.4 ms 降到 1.7 ms；消除 65.5K 的断崖；decode 桶覆盖到 262K |
| **decode 输入准备向量化**：GDN 状态索引和注意力组页表从 Python 循环改为 numpy，上线时先与原实现逐轮比对 | `VLLM_HPU_FAST_DECODE_PREP=all` | 200K 上下文时页表构建 43.6 → 1.9 ms；与下两项一起使 2 路并发每轮约 64 → 52 ms |
| 并发上限 4 → 2，环境变量读取缓存 | `--max-num-seqs 2`、`VLLM_HPU_ENVS_CACHE=1` | 同上（三项未分别消融） |
| 修复候选选择器断图 | 无（代码修复） | 每轮 10 个碎片帧消失；整轮耗时无可测变化 |
| 修复 2 路并发时 drafter 注意力断图 | `VLLM_HPU_SKIP_UNBACKED_BLOCK_LIST=1` | drafter 阶段主机耗时降到与单路相同；整轮耗时无可测变化 |
| 注意力组共用掩码 | `VLLM_HPU_SHARE_GROUP_MASKS=1` | 每轮少约 18 个零散设备算子；整轮耗时无可测变化 |
| B≤2 的 GDN prepared 核路径 | `FLASHINFER_GAUDI_MTP_PREPARED_MAX_BATCH=2` | B1/B2 数值测试通过；2 路每轮改善约 5–8% |

后三行如实写了「无可测变化」：修复本身生效，但没有换来速度。原因分析和其它没有收益的尝试见 [优化说明第 5 节](docs/OPTIMIZATION.md#5-没有带来收益或没有证实的事项)。

<details>
<summary><b>更早的优化与历史测速记录</b>（v21 – v25，展开查看）</summary>

以下是更早的本地历史记录，不是当前配置的统一 benchmark。

| 项目 | 实际改动 | 状态与效果边界 |
|---|---|---|
| 非 greedy 投机解码 | 移植/适配本地 MTP 路径，接通 DFlash2 的目标验证、接受判断、拒绝纠正和状态回滚；保留主模型随机采样 | 已有非 greedy 短文本生成验收；不把草稿的确定性选词误称为主模型 greedy。完整模型概率/质量等价性仍需验证 |
| 随机拒绝纠正的 HPU 实现 | 增加 TPC 随机纠正/输出打包扩展与 Python 接线，减少纠正阶段的小算子及 host 往返 | 源码见 [native 采样扩展](native/mtp-sampling/) 和 [纠正入口](vllm-gaudi/vllm_gaudi/v1/sample/native_correction.py)；没有单独消融得到可归因的整体提速百分比 |
| DFlash 辅助特征感知分组 | 按辅助特征捕获边界合并目标 decoder 的编译区域；将 64 层组织为 19 组，仍完整执行每一层并保留 5 个捕获点 | 有短请求热性能改善，见下表；没有跳层、降低精度或改成 greedy |
| CPU 交接与采样参数准备 | token list 交接、标量选择/采样准备缓存，减少部分重复准备 | 同进程热对照约 1% 变化，尚未证明稳定收益；小拷贝打包未启用 |
| Dynamo 静态 guard 筛选 | 对限定的参数、空 hook 和模块重复关系筛选 guard；保留输入、常量和运行时状态相关检查 | 已有 guard 回归；没有得到可测的收益 |

**DFlash2 单流短请求。** 条件：43-token 故事 prompt、128 输出 token、`temperature=0.7`、`top_p=0.9`、`top_k=20`、`seed=123`、关闭 thinking、7 个草稿 token、无 profiler。这里的 token/s 是 128 ÷ 请求总时间，包含首 token 前时间，不是纯 decode 速度。

| 改动 / 对照 | 优化前 | 优化后 | 如何解读 |
|---|---|---|---|
| 辅助特征感知分组，v21 → v22 | 6.913 秒 / 18.52 token/s | 4.160 秒 / 30.77 token/s | 历史热请求耗时下降约 39.8%；跨进程、少量样本，固定 seed 文本不同，不能称为严格数值等价的普遍收益 |
| CPU 交接＋采样准备，v25 同进程最后热对照 | 3.968 秒 / 32.26 token/s | 3.931 秒 / 32.57 token/s | 耗时下降约 0.95%；只有一组最终热对照，约 1% 差异可能是波动 |

v22/v25 的该短故事接受率约 6.81%（41/602），86 轮输出 128 token，约 1.49 输出 token/轮。分组主要降低每轮执行开销，没有让草稿更准。历史同 prompt 的 MTP-7 热请求约 3.855 秒 / 33.20 token/s，因此不能声称 DFlash2 在这个负载上已经优于 MTP。

冷请求必须单独报告：v22 首条约 6.488 秒，v25 baseline 首条约 6.668 秒；不能把冷请求与后续热请求直接相除当作补丁收益。

**预录候选：正确，但没有净提速。** 这是独立的离线实验，候选和私有框架不在本仓库中，也没有上线。真实前 8 层（6 层 GDN＋2 层注意力）、完整 48 组状态池、B1/T8，接受长度 1/4/8 的输出与全部缓存严格匹配普通组。

| 完整状态池下的热组测试 | 普通执行 | 预录执行 |
|---|---:|---:|
| 提交 wall 中位数 | 0.923 ms | 0.862 ms |
| 整组完成 wall 中位数 | 5.613 ms | 5.797 ms |

提交减少约 6.7%，但完成时间增加约 3.3%，所以不作为加速方案上线。这些是组级 wall 时间，不是 HPU kernel 时间或模型 token/s。

</details>

## 仓库结构

```text
vllm-core/            配套 vLLM 主程序源码，叠加服务器实际安装的 Python 文件
vllm-gaudi/           Gaudi 插件、FlashInfer-Gaudi、C++/TPC 核及相关测试
native/mtp-sampling/  随机拒绝纠正 / 输出打包扩展源码
validation/           不依赖设备的离线测试（2504 项）和断图复现脚本
tools/                测速脚本与 trace 分析脚本
examples/             本次实测所用配置的启动脚本示例
docs/                 优化说明、快照说明、图表
SOURCE_SNAPSHOT.json  源码清单、SHA-256 与来源版本
```

`vllm-core/`、`vllm-gaudi/`、`native/` 是服务器源码快照，由 `SOURCE_SNAPSHOT.json` 逐文件记录哈希。`validation/`、`tools/`、`examples/` 和 `docs/` 里新写的文件不属于服务器源码，不在清单内。

## 快速上手

**跑离线测试**（不需要 Gaudi 设备，只要 torch、numpy、pytest）：

```bash
python -m pytest -q validation
```

**看配置怎么写**：[`examples/launcher-v5.example.sh`](examples/launcher-v5.example.sh) 是本次实测所用的环境变量和启动参数，路径需要换成你自己的。

**测速和采 trace**：见 [`tools/README.md`](tools/README.md)，里面有采集方法和几个容易踩的坑（比如硬件 trace 默认开启，采整个请求会把内存撑到 30 GiB 以上）。

**改代码后的验证流程**（实践下来比较省时间的顺序）：

1. 离线测试通过。
2. 加 `VLLM_SKIP_WARMUP=true` 启动，约 6 分钟可用，先确认能编译、能跑、输出正常。
3. 再带完整预热启动（约 30 分钟），然后才测速。

## 本地适配覆盖的方向

- **Qwen3.5 / Qwen3-Next 混合架构**：完整注意力、Gated DeltaNet、卷积历史及投机验证检查点的 HPU 路径。
- **FP8 推理**：Gaudi2 权重范围/布局处理、动态量化与 FP8 GEMM。模型 FP8 线性不等于所有算子或 KV 缓存都使用 FP8。
- **DFlash2 与 MTP**：草稿生成、目标验证、状态回滚和非 greedy 随机拒绝纠正。不同模型、投机宽度与 batch 的资格并不相同，不能任意互换。
- **主机提交链路**：decode 输入准备、注意力组路由、共享前缀验证、环境变量缓存。
- **OpenAI 兼容服务**：沿用 vLLM API；自动工具调用需启用相应选项并选择与模型模板匹配的 parser，本地配置使用 `qwen3_xml`。
- **诊断入口**：有界计时和 profiling 支持。诊断配置不是性能配置。

FP8 GEMM、基础 RMSNorm/RoPE 和部分 GDN 核来自原有 Gaudi/1Cat 实现。本项目对这些路径做适配与接线，不把已有的上游能力算作新写的优化。

## 软件栈

| 组件 | 版本 / 基线 |
|---|---|
| 硬件 | Intel Gaudi2，单卡，TP=1 / PP=1 |
| Python | 3.12 |
| Gaudi 驱动 | `1.24.1-b336d5e` |
| Habana Torch plugin | `1.24.1.482` |
| PyTorch | `2.11.0a0+git009b5f6` |
| vLLM | `0.28.1rc1.dev100+gfe755c889.d20261009.empty` |
| vLLM 核心 Git 基线 | `fe755c88995ad468882517b6c4bdd60138d46a3a` |
| Gaudi 插件 Git 基线 | `57a50be3f7fc3a61c0a89c9ae3a3612072269b99` |

Git 基线不包含全部本地改动，请以本仓库文件和 `SOURCE_SNAPSHOT.json` 为准。

执行方式是 **non-lazy / eager 加区域编译**。`torch.compile`、Native Replay 和 HPUGraph 是不同机制，不能统称「已开启 Graph」。PyTorch、Habana 插件、Synapse 与自定义 native 库的版本和 ABI 必须匹配。

## 使用前须知

1. 准备匹配的 Gaudi 软件栈及 SDK。不要为安装本项目随意刷固件、替换驱动，或让通用 CUDA 安装命令覆盖 HPU 版 PyTorch。
2. 分别阅读 [vLLM 核心说明](vllm-core/README.md) 和 [Gaudi 组件构建配置](vllm-gaudi/pyproject.toml)。这是多组件快照，仓库根目录不能作为单一 Python 包安装。
3. 自定义 FlashInfer-Gaudi/TPC 算子及 [随机采样扩展](native/mtp-sampling/) 需要按目标环境构建。仓库不包含编译出的 `.so`、wheel 或 SDK 二进制。
4. 设置本机的模型目录、动态库路径、kernel 搜索路径及日志目录。源码中保留的部分默认路径来自原环境。
5. 先做组件数值验证，再做非 greedy 的短请求、工具调用和流式验收，最后比较同模型、同参数的热请求基线。不要跳过预热后报告峰值。

参考 [Intel Gaudi 文档](https://docs.habana.ai/en/latest/index.html) 与 [vLLM Gaudi 文档](https://docs.vllm.ai/projects/gaudi/en/0.26.0/index.html)；后者是单独的文档版本，不等于本快照的源码基线。

## 验证边界

已经做的：

- 离线 2504 项 CPU 测试，被测代码直接取自本仓库源码。
- 真机冒烟加完整预热，117 个 prompt 形状和 45 个 decode 形状全部编译通过。
- 2 路并发下，新开关全开与全关的输出 800 个 token 逐字相同。
- 单路 0.6K 到 86K 上下文的测速，以及 30K 的设备 trace 对照。

还没做或做不了的：

- **256K 没有实测**，只有按线性规律的外推（约 100 ms/轮）。
- 单路 `temperature=0` 的输出在同一个服务上重跑也不逐字一致，原因未查，所以单路没有逐字对照。
- 部署的请求守卫拒绝 logprobs，没有逐 token 的数值比对。
- 工具调用、流式中途取消、抢占、多日稳定性没有在新配置下重新验收。
- prefill（首 token 时间）没有优化过：86K 提示词约 32 秒，256K 外推约 3.5 分钟。
- 测速用的是带 profiler 的测试构建，正式配置下的数字可能略有不同。
- 没有宣称达到硬件性能上限，也没有承诺投机解码必然优于普通解码。

## 项目来源与许可

本项目基于 1Cat-vLLM-Gaudi 工程分支、[vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) 硬件插件及 [vllm-project/vllm](https://github.com/vllm-project/vllm) 推理引擎。这是一次源码发布，不是向上游提交的 PR。部分本地适配、测试和文档使用了 AI 辅助，仍需要人工审查与目标环境验收。

主体来源项目采用 Apache-2.0，分别见 [vLLM LICENSE](vllm-core/LICENSE) 与 [Gaudi 插件 LICENSE](vllm-gaudi/LICENSE)。第三方目录和文件保留各自许可，不将其统一重授为其他许可。

发布内容包含公开源代码、原始测试/构建入口，以及服务器新增的源文件；排除模型权重、缓存、编译产物、运行日志、trace、备份副本、机器凭据、私有环境文件和原仓库的 CI workflow。源码校验方法：按 `SOURCE_SNAPSHOT.json` 中的相对路径读取文件并计算 SHA-256。
