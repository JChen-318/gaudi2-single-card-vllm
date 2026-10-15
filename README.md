<div align="center">

# Gaudi2 Single-Card vLLM

**面向单张 Intel Gaudi2 的大模型推理适配与优化**

FP8 · 非 greedy 投机解码 · 长上下文 · OpenAI 兼容接口

![Gaudi2](https://img.shields.io/badge/Intel%20Gaudi2-single%20card-0071c5)
![License](https://img.shields.io/badge/license-Apache--2.0-2ea44f)

[项目特点](#项目特点) · [性能概览](#性能概览) · [快速开始](#快速开始) · [配置与运维](#配置与运维) · [文档](#文档)

</div>

把 vLLM、Gaudi 硬件插件和本地算子整合成一套单卡推理工程，让一张拥有 96 GiB HBM 的 Gaudi2 运行 27B 级模型，并改善日常对话、代码生成和长文本处理的体验。

项目重点围绕 **Qwen3.8 的 27B FP8 模型**，提供 DFlash2 和 MTP 的本地适配，保留随机采样，并针对单卡环境优化解码链路。仓库包含完整源码、一键安装与启动脚本、离线验证和测速工具：在同样的 Gaudi 软件环境下，按「快速开始」几条命令即可部署。

## 项目特点

- **单卡 FP8 推理**：针对 Gaudi2 的权重布局、混合注意力和 GDN 状态管理进行适配。
- **非 greedy 投机解码**：支持本地 DFlash2/MTP 路径，保留 temperature、top-p、top-k 采样，不以切换 greedy 换取速度。
- **长上下文优化**：验证 token 共用历史 KV 前缀，减少重复读取；扩展解码分桶覆盖范围。
- **主机与设备流水重叠**：按预测提前提交下一轮主模型和采样，紧接着用设备上的采样结果提交该轮草稿，不等主机读回；下一步核对一致后直接复用，下一轮输入增量生成，两路并发同样适用。
- **更轻的主机准备**：向量化生成索引和页表，编译区域独立代码对象、草稿模型整体编译，减少缓存查找与零散提交。
- **更少的设备时间**：采样和草稿候选不再对整个词表排序，词表投影使用 FP8，GDN 检查点直接写入状态池；重写的 [GDN 递推 TPC kernel](native/gdn-fused-kernel/) 把准备步骤并入 kernel，跨通道归约共用一棵归约树。
- **FP8 KV 缓存**：同样的显存放下 52.7 万 token（两个 256K 请求），长上下文每轮更快。
- **专用采样扩展**：提供 HPU 随机拒绝纠正与输出打包源码，衔接投机验证后的生成流程。
- **编译结果缓存与更短的预热**：单文件 TPC 内核库转接层让多个内核库可以配合 `PT_HPU_RECIPE_CACHE_CONFIG` 使用；预热不再在每个分桶前后强制 gc，从 48 分钟缩短到 23 分钟。
- **熟悉的服务接口**：沿用 vLLM 的 OpenAI 兼容接口，提供工具调用配置示例。

## 性能概览

以下为本地单张 Gaudi2、27B FP8 目标模型搭配 DFlash2 的测量记录（v11 生产配置），使用非 greedy 采样、关闭 thinking，每条生成 512 token。

| 场景 | 解码吞吐 |
|---|---:|
| 单路中英文解释、议论文、科普 | 81–111 token/s |
| 单路代码生成（Python / Go / TypeScript） | 164–192 token/s |
| 单路结构化输出、复述类 | 265–278 token/s |
| 两路代码与推理，合计 | 264–312 token/s |
| 单路约 3 万 token 上下文 | 约 79 token/s |
| 单路约 8.6 万 token 上下文（v5 时测量） | 47.6 token/s |

表中吞吐来自 v11。每轮验证的耗时在各类提示词上都稳定在 27.6–28.1 ms（单路短上下文；v10 为 30.5–33 ms，v5 为 42.3 ms），两路并发每轮 31.9 ms（v10 为 35.0 ms，v5 为 52.2 ms），3 万 token 上下文每轮 33.7 ms（v10 为 37.3 ms，v5 为 48–51 ms）。当前 v14 生产配置（GDN kernel 重写 + FP8 KV 缓存）进一步降到单路 26.2–26.3 ms、两路 29.7 ms、3 万 token 30.2 ms，约 10 万 token 上下文每轮 38.5 ms（约 75 token/s）。不同内容的速度差别主要来自草稿接受率：代码和结构化输出每轮可接受 3.6–6.8 个草稿 token，自然语言约 1.3–2.2 个，中文散文类创作更低。完整测试条件、逐项拆分和前后对照见 [优化说明](docs/OPTIMIZATION.md)。

这些数字采用流式 decode 口径，不是整请求吞吐；任务、草稿接受率和硬件环境都会影响结果。**上下文配置上限为单请求 262,144 token（FP8 KV 缓存可同时容纳两个），当前最长实测约 10 万 token**，不把外推结果当成实测成绩。

## 快速开始

### 需要准备

| 项 | 要求 |
|---|---|
| 加速卡 | 1 张 Intel Gaudi2（HL-225，96 GiB HBM） |
| 主机 | x86-64，内存约 32 GiB 及以上（加载权重时要经过主机内存），模型所在磁盘约 31 GiB |
| 系统 | Ubuntu 24.04，Python 3.12 |
| Gaudi 软件 | SDK **1.24.1**：驱动、固件、`hl-smi`、TPC 编译器 `tpc-clang`、`habanalabs-graph` 头文件 |
| PyTorch | Intel 为 1.24.1 发布的 PyTorch 包（`torch 2.11.0a0`、`habana-torch-plugin 1.24.1.482`） |
| 构建工具 | `cmake`、`g++`、`objcopy`、`libtcmalloc.so.4` |
| 模型 | Qwen3.8-27B 的 FP8 权重（含 MTP 层），以及对应的 DFlash2 草稿模型 |

Gaudi 驱动和 PyTorch 的安装见 [安装说明](docs/INSTALL.md)。仓库不包含编译好的库和模型权重，下面的脚本会在本机编译。

### 安装与启动

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-vllm.git
cd gaudi2-single-card-vllm

scripts/check_env.sh                 # 检查驱动、编译器、Python 与 PyTorch 版本（只读，不占用设备）
scripts/install.sh --venv .venv      # 安装 vllm-core 与 vllm-gaudi（可编辑模式，运行的就是仓库里的源码）
scripts/build_native.sh              # 编译采样内核、FlashInfer-Gaudi 内核和内核库转接层，约 2 分钟

cp config/serve.env.example config/serve.env
vi config/serve.env                  # 至少填写 TARGET_MODEL 和 DRAFT_MODEL

scripts/serve.sh --daemon            # 后台启动，日志在 ~/.cache/gaudi2-single-card-vllm/logs/
scripts/smoke_test.sh --wait         # 等服务就绪后发几条请求，检查输出和速度
```

**第一次启动需要 30–45 分钟**：加载模型约 4 分钟，然后预热，把每种形状的计算图编译一遍。设备端的编译结果会缓存到磁盘，之后每次重启约 25 分钟。只想先确认能跑通，可以用 `scripts/serve.sh --skip-warmup`，约 6 分钟就能接受请求，但每种新形状第一次出现时都要现场编译，这种模式不要拿来测速。

停止服务：`scripts/stop.sh`。

服务需要锁定较多主机内存，实测以 root 运行（`sudo scripts/serve.sh --daemon`，此时请在 `serve.env` 里显式设置 `DATA_DIR`）。用普通用户运行时，要把该用户的 memlock 上限设为 unlimited。

### 调用

服务提供 OpenAI 兼容接口，默认只监听本机 `127.0.0.1:8000`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-fp8-mtp",
  "messages": [{"role": "user", "content": "用 Python 写一个快速排序"}],
  "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(model="qwen3.8-27b-fp8-mtp",
                                   messages=[{"role": "user", "content": "你好"}], max_tokens=256)
print(r.choices[0].message.content)
```

- 默认采样参数 `temperature=1.0, top_p=0.95, top_k=20`，请求里可以覆盖。思考模式由 `chat_template_kwargs.enable_thinking` 控制。
- 支持流式输出和工具调用，`tool_choice` 只能是 `"auto"`。投机解码路径不支持以下请求，服务会直接返回错误：强制工具调用、结构化输出（`response_format` / guided decoding）、各类惩罚、`min_p` / `min_tokens`、`logit_bias`、`logprobs`、思考预算。
- 对外提供服务时，在 `config/serve.env` 里设置 `API_KEY`，`HOST` 改为 `0.0.0.0`，并放在防火墙或反向代理之后。

### 测速

```bash
scripts/bench.sh          # 单路 3 条、两路并发、约 3 万 token 上下文，约 5 分钟
scripts/bench.sh --full   # 再加 15 类提示词逐条测速
```

速度主要取决于每轮能接受几个草稿 token，所以会随提示词内容和随机采样变化。

## 配置与运维

- 所有设置都在 `config/serve.env` 里：模型路径、监听地址、API key、上下文长度、并发数、KV 缓存大小、绑核。调优用的开关都写在 `scripts/serve.sh` 中，每个开关的含义见 [配置说明](docs/CONFIG.md)。
- 缓存、日志和运行期控制文件放在 `DATA_DIR`（默认 `~/.cache/gaudi2-single-card-vllm`）。设备端编译缓存的目录名里带着各内核库的哈希，重新编译内核库后会自动换用新的缓存目录，旧目录可以手动删除。
- 更新代码后重新运行 `scripts/install.sh`（依赖有变化时）和 `scripts/build_native.sh`（`native/` 或 `vllm-gaudi/csrc/` 有变化时），然后重启服务。
- 以 systemd 服务运行、调整并发数或上下文长度，见 [配置说明](docs/CONFIG.md)。

## 仓库结构

```text
scripts/             安装、编译、启动、停止、冒烟测试、测速脚本
config/              服务配置模板 serve.env.example
requirements/        与实测环境一致的 Python 依赖版本
vllm-core/           配套 vLLM 主程序
vllm-gaudi/          Gaudi 插件、FlashInfer-Gaudi（Python 包与 C++/TPC 源码）与设备算子
native/mtp-sampling/ 随机拒绝采样的 TPC 内核与 PyTorch 算子
native/tpc-kernel-shim/ 单文件 TPC 内核库转接层（让编译缓存能同时加载多个内核库）
native/gdn-fused-kernel/ 融合 GDN 递推 kernel 的说明、模拟器测试与真机校验脚本
validation/          离线验证（CPU 上运行）
tools/               测速和 trace 分析工具
examples/            历史版本的独立启动脚本，作对照用
docs/                安装、配置、优化记录与源码来源说明
```

## 文档

- [安装说明](docs/INSTALL.md)：Gaudi 驱动与 PyTorch、依赖、常见安装问题。
- [配置说明](docs/CONFIG.md)：`serve.env` 各项、`serve.sh` 中每个调优开关、systemd、改并发与上下文。
- [优化说明与实测](docs/OPTIMIZATION.md)：具体改动、效果和仍待验证的方向。
- [离线验证](validation/README.md)：测试范围与运行方法。
- [测速与分析工具](tools/README.md)：性能测量和诊断入口。
- [源码来源说明](docs/SNAPSHOT.md)：各组件来源与文件校验方式。

## 已知限制

- 只针对单卡、一到两路并发调优和实测。`MAX_NUM_SEQS` 调大可以运行，但预热分桶和速度都没有验证。
- prefill（首 token 时间）没有专门优化：3 万 token 的提示词首 token 约 7 秒。
- FP8 KV 缓存没有做校准，接受率和抽样输出正常，但没有做系统的质量评测。FP8 词表投影与 BF16 的 top-1 一致率 98.4%。
- 只用上文这两个模型测试过。其他 Qwen3.5 架构的模型或草稿模型需要自行验证。
- 投机解码的收益取决于内容：代码和结构化文本提速明显，中文散文类创作接受率较低。
- 工具调用、取消/抢占和 10 万 token 以上的上下文还需要更多验收。

## 来源与许可

基于 1Cat-vLLM-Gaudi 工程分支、[vLLM Gaudi 插件](https://github.com/vllm-project/vllm-gaudi) 和 [vLLM](https://github.com/vllm-project/vllm)，不是 Intel 或 vLLM 的官方发行版。基础 FP8、归一化、位置编码及部分 GDN 能力沿用原有实现，本项目在其上进行单卡适配与优化。

主体采用 [Apache-2.0](LICENSE)，第三方组件保留各自许可。部分适配、测试和文档使用了 AI 辅助，仍需人工审查与目标环境验证。
