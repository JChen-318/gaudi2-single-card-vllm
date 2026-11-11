# 直接编程 MME/TPC 的注意力（MiniMax H3 实验路径）

绕过图编译器，用 SCAL 私有流直接提交 Gaudi2 的 MME 描述符和自写 TPC 内核，为 MiniMax H3 DiT 的长序列注意力（56 头 × 37,760 token × D=128）做的一条可选实验路径。

**生产默认启动不使用它**：它只通过隔离容器里的 `sitecustomize` 覆盖层（`attention/h3inject/`）按环境变量开启，vLLM / vLLM-Omni 源码不改。精度门限和视频结果见下文；它还不是经过生产合格认证的替换。

## 结果（单卡 Gaudi2，5 秒 1344×768，4 步 LightX2V Turbo，稳态请求）

| 配置 | 单次请求 |
|---|---:|
| 生产（`docs/OPTIMIZATION.md` 23.9，同环境复测） | 100.45 s |
| direct 注意力（流式 head + 编译门控） | 84.41 s |
| 再加文本编码器锁页 + t2va 时 DiT 常驻（VAE tile 4） | **79.7 s** |

- 单层注意力（56 头，真实 q/k/v）：生产 323 ms → 233.7 ms；单头 3.93–4.13 ms，约 12.7–13 µs / 128 行块。
- 精度：逐层 FP32 采样（188 次调用 / 1,788 个 head）最大 RMS 0.78%，门限 1%；不过门的 head（含 NaN/Inf、范数上界过大）逐个退回原路径，45/48/49 三层整层保留 FusedSDPA。
- 视频：与生产输出 PSNR 约 18 dB；但生产本身与全 FusedSDPA 也只有约 18 dB，扩散过程会放大注意力实现间的微小差异，所以这不能单独当作质量回退的证据，也不能证明质量合格。
- 锁页 / DiT 常驻两项不改变 DiT 输出（音频逐位相同）；视频只因 VAE tile 批 4（对 28）不同，而 tile 4 与逐块参考逐位相同。DiT 常驻只适用 t2va/FL2VA（Ref2VA 需要 phase offload）。

## 适用范围与回退（issue #3 之后，2026-11-11）

- H3 attention 长度 = 视频 37296 + 音频 414 + 文本 token 数。上面的结果是 40 个文本 token（L=37750）。6da6fb1 之前的版本只接受 L=37750，换一个长度不同的提示词就整层回退，没有加速。
- 现在支持任意 3·FIRST < L ≤ KEYS。长度变化时只重建 TPC exp/归一化程序并原地改写显存里的代码，MME 程序和 56 套 head 缓存不变，超出长度的 Q/K/V 尾部清零。
- 容量档 `H3DIRECT_KEYS`：
  - `37760`（默认）：文本 ≤ 50 token，输出与之前逐位相同。
  - `38272`：文本 ≤ 562 token，attention 约多 1% 工作量。每个进程只有一个执行计划，提示词长度不一的服务应选这一档。
  - 超出容量的长度仍走原路径。
- 过不了门限的 head 改走带真实行最大值的精确 attention（`exact_attention`），不再交给 MAXFREE。生产 MAXFREE 本身也加了运行时下溢检测：构造的反平行/正交大范数输入原来输出全 0，现在输出 1；真实层输出逐位不变，每层多约 0.3 ms。
- 证据：`attention/evidence/issue3-fix-20261111.md`；测试 `attention/issue3_test.py`。

## 硬件结论（本软件栈 SynapseAI 1.24.1）

- **QK 受每个 MME 的输出侧限制**：恒定输出、K 从 32 到 256 时间几乎不变；单 MME 写一半输出与双 MME 写全部同时。斜率约 0.82 TB/s/MME（双 MME 约 1.65 TB/s），原版编译器的 `torch.matmul` 也到同一值（`attention/evidence/mme-compiler-reference-20261109.md`、`mme-ksweep-20261109.md`）。这不是写活跃计数器的测量，外部"单 MME 3.2 TB/s"的口径未能复现。
- **PV 受输入读取限制**：每个 P 元素只复用 D=128 次，原版编译器与本实现都约 210 TFLOPS（峰值约 48%），转置成宽输出也一样。
- **同一 MME 上 QK 与 PV 不重叠**，有无 barrier、先后顺序都一样（`mme-qkpv-overlap-20261109.md`）。
- 因此每块 MME 下限约 11.8 µs，MME-only 流水实测 11.60 µs，已在下限；剩余约 1.1 µs/块是 TPC/MME 争用和同步。
- 使用的是两个逻辑 MME 的 master+slave 全阵列（描述符保留 slave；`--slave drop` 会超时）。

## 主要做法

- 不求行最大值的有界 exp：CPU 用 q/k 范数证明上界后，TPC 直接查表求 exp（FP16 分数、BF16 概率），分母四段精确合并。
- 编译器描述符重放：`attention/mme_dump.py` 把 `PRINT_MME_DESCRIPTORS` 的转储转成 182 个寄存器字，只重定位 A/B/C 地址；`fixtures/*/mme_debug` 是这些转储。
- 1024 行一次 MME 提交（M512 描述符行循环重复两次）、跨组流水、多写 monitor、分离 fence、56 套 head 程序缓存、head 间只做原生 DMA 与提交。
- H3 集成：`attention/h3_direct_attention.py` + `attention/h3inject/sitecustomize.py`，开关 `H3DIRECT_ENABLE`、`H3DIRECT_POW2Q`、`H3DIRECT_STREAM`、`H3DIRECT_FAST_GATES`、`H3DIRECT_COMPILED_GATES`；视频 A/B 脚本 `attention/h3_direct_video_run.sh`（另有 `H3_TE_PIN`、`H3_PHASE_OFFLOAD`、`H3_VAE_TILE_BATCH`、`H3_TRACE_CALL`、`H3_TRACE_DECODE`）。

## 目录

| 路径 | 内容 |
|---|---|
| `attention/` | 运行时（`context_bridge.cpp`、`runtime_*.hpp`）、TPC 内核（`kernels/`）、探针与基准脚本、H3 集成、`README.md`（详细构建与复现步骤） |
| `attention/evidence/` | 各项结论的摘要（md/json）；原始日志、trace、视频不在仓库 |
| `attention/fixtures/*/mme_debug/` | 编译器 MME 描述符文本转储 |
| `attention/third_party/` | Habana synapseai-core 寄存器 / LUT 头（MIT，含 `COPYING.MIT`）与 HCL SCAL 包头（上游 Apache-2.0） |
| `direct-mme/` | 最早的单 master GEMM 直接提交原型 |

脚本里的服务器路径（`/home/a/...`）与模组号（7）是作者机器上的值，复现时需替换。真实 q/k/v 输入、构建产物、失败实验和服务器访问工具没有包含。
