<div align="center">

# Gaudi2 Single-Card vLLM

**面向单张 Intel Gaudi2 的大模型推理适配与优化**

FP8 · 非 greedy 投机解码 · 长上下文 · OpenAI 兼容接口

![Gaudi2](https://img.shields.io/badge/Intel%20Gaudi2-single%20card-0071c5)
![License](https://img.shields.io/badge/license-Apache--2.0-2ea44f)

[项目特点](#项目特点) · [性能概览](#性能概览) · [开始使用](#开始使用) · [文档](#文档)

</div>

把 vLLM、Gaudi 硬件插件和本地算子整合成一套单卡推理工程，让一张拥有 96 GiB HBM 的 Gaudi2 运行 27B 级模型，并改善日常对话、代码生成和长文本处理的体验。

项目重点围绕 **Qwen3.8 的 27B FP8 模型**，提供 DFlash2 和 MTP 的本地适配，保留随机采样，并针对单卡环境优化解码链路。仓库包含配套源码、离线验证、测速工具和启动示例，方便继续开发或在相近环境中复现。

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

## 开始使用

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-vllm.git
cd gaudi2-single-card-vllm
```

1. 准备版本匹配的 Gaudi 软件栈和 HPU 版 PyTorch，参考 [Gaudi 插件说明](vllm-gaudi/README.md) 与 [vLLM 主程序说明](vllm-core/README.md)。
2. 按目标 SDK 构建需要的算子、[本地采样扩展](native/mtp-sampling/) 和 [内核库转接层](native/tpc-kernel-shim/)（使用编译缓存时需要）。仓库不包含编译好的动态库或模型权重。
3. 修改 [启动示例](examples/launcher-v14.example.sh) 中的模型、库和日志路径，先验证短请求，再进行完整预热与测速。较早的 [v11](examples/launcher-v11.example.sh)、[v10](examples/launcher-v10.example.sh) 和 [v5](examples/launcher-v5.example.sh) 示例保留作对照。

这是多组件源码工程，不能直接在仓库根目录 `pip install`。示例来自特定部署环境，需要按本机配置调整；诊断和跳过预热选项不应作为正式测速配置。

## 仓库结构

```text
vllm-core/           配套 vLLM 主程序
vllm-gaudi/          Gaudi 插件、FlashInfer-Gaudi 与设备算子
native/mtp-sampling/ 本地随机采样扩展
native/tpc-kernel-shim/ 单文件 TPC 内核库转接层
native/gdn-fused-kernel/ 融合 GDN 递推 TPC kernel、模拟器测试与 FlashInfer-Gaudi 补丁
examples/           启动配置示例
validation/         离线验证与编译问题复现
tools/              测速和 trace 分析工具
docs/               优化说明、测试记录与快照说明
```

## 文档

- [优化说明与实测](docs/OPTIMIZATION.md)：具体改动、效果和仍待验证的方向。
- [离线验证](validation/README.md)：测试范围与运行方法。
- [测速与分析工具](tools/README.md)：性能测量和诊断入口。
- [源码快照说明](docs/SNAPSHOT.md)：组件来源与文件校验方式。

项目仍在持续优化，重点验证单卡、一到两路并发的场景；最新配置的工具调用、取消/抢占和更长上下文还需要进一步验收。FP8 词表投影会轻微改变 logits（与 BF16 的 top-1 一致率 98.4%），它对草稿接受率的影响还没有做开/关对照。投机解码的收益取决于负载，不保证所有任务都比普通解码更快。

## 来源与许可

基于 1Cat-vLLM-Gaudi 工程分支、[vLLM Gaudi 插件](https://github.com/vllm-project/vllm-gaudi) 和 [vLLM](https://github.com/vllm-project/vllm)，不是 Intel 或 vLLM 的官方发行版。基础 FP8、归一化、位置编码及部分 GDN 能力沿用原有实现，本项目在其上进行单卡适配与优化。

主体采用 [Apache-2.0](LICENSE)，第三方组件保留各自许可。部分适配、测试和文档使用了 AI 辅助，仍需人工审查与目标环境验证。
