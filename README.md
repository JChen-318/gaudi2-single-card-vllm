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
- **更轻的主机准备**：向量化生成索引和页表，优化层分组、GDN 核分发及编译边界，减少重复准备与零散提交。
- **专用采样扩展**：提供 HPU 随机拒绝纠正与输出打包源码，衔接投机验证后的生成流程。
- **熟悉的服务接口**：沿用 vLLM 的 OpenAI 兼容接口，提供工具调用配置示例。

## 性能概览

以下为本地单张 Gaudi2、27B FP8 目标模型搭配 DFlash2 的测量记录，使用非 greedy 采样、关闭 thinking。

| 场景 | 解码吞吐 |
|---|---:|
| 单路短上下文对话 | 67–75 token/s |
| 单路代码生成 | 112–134 token/s |
| 两路代码生成，合计 | 约 209 token/s |
| 单路约 3 万 token 上下文 | 59–65 token/s |
| 单路约 8.6 万 token 上下文 | 47.6 token/s |

长上下文是近期优化的重点：共享前缀后，约 3 万 token 的记录从 45.2 提升到 59–65 token/s；更长上下文也有明显改善。完整测试条件和前后对照见 [优化说明](docs/OPTIMIZATION.md)。

这些数字采用流式 decode 口径，不是整请求吞吐；任务、草稿接受率和硬件环境都会影响结果。**上下文配置上限为 262,144 token，当前最长实测约 8.6 万 token**，不把外推结果当成实测成绩。

## 开始使用

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-vllm.git
cd gaudi2-single-card-vllm
```

1. 准备版本匹配的 Gaudi 软件栈和 HPU 版 PyTorch，参考 [Gaudi 插件说明](vllm-gaudi/README.md) 与 [vLLM 主程序说明](vllm-core/README.md)。
2. 按目标 SDK 构建需要的算子和 [本地采样扩展](native/mtp-sampling/)。仓库不包含编译好的动态库或模型权重。
3. 修改 [启动示例](examples/launcher-v5.example.sh) 中的模型、库和日志路径，先验证短请求，再进行完整预热与测速。

这是多组件源码工程，不能直接在仓库根目录 `pip install`。示例来自特定部署环境，需要按本机配置调整；诊断和跳过预热选项不应作为正式测速配置。

## 仓库结构

```text
vllm-core/           配套 vLLM 主程序
vllm-gaudi/          Gaudi 插件、FlashInfer-Gaudi 与设备算子
native/mtp-sampling/ 本地随机采样扩展
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

项目仍在持续优化，重点验证单卡、一到两路并发的场景；最新配置的工具调用、取消/抢占和更长上下文还需要进一步验收。投机解码的收益取决于负载，不保证所有任务都比普通解码更快。

## 来源与许可

基于 1Cat-vLLM-Gaudi 工程分支、[vLLM Gaudi 插件](https://github.com/vllm-project/vllm-gaudi) 和 [vLLM](https://github.com/vllm-project/vllm)，不是 Intel 或 vLLM 的官方发行版。基础 FP8、归一化、位置编码及部分 GDN 能力沿用原有实现，本项目在其上进行单卡适配与优化。

主体采用 [Apache-2.0](LICENSE)，第三方组件保留各自许可。部分适配、测试和文档使用了 AI 辅助，仍需人工审查与目标环境验证。
