# Gaudi2 SRAM 注意力原型

2026-11-08。主程序在独立设备上下文中直接执行：

`MME(QK) → TPC(exp、行和) → MME(PV) → TPC(归一化)`。

S 与 P 共用 SRAM 存储，全链路不把这两个中间矩阵写入 HBM。
运行程序只依赖 hl-thunk；描述符源自已保存的编译器导出，扩展字段经 256/1024 维导出交叉核对。

## MME读写模型核查（2026-11-09）

当前1.24.1 BF16 GEMM CPU查询：单逻辑MME读A/B各4端口、写C4端口；整片分别8/8/8。HAL getter返回整数200。已追到Gaudi3带宽属性生产与通用编译器消费者，直接乘端口并按GB/s输出，无时钟换算，支持软件单位GB/s。**Gaudi2的calcMemoryAttributes在本构建为空实现，因此200×端口仅为条件算术，不能当已证实的Gaudi2模型预算或物理上限。** 原版编译器整片写出实测1.638/1.668TB/s仍有效，不能用低于峰值的实测排除字节/周期假说，也没有证伪外部单MME→SRAM3.2TB/s。完整证据 `evidence/mme-bw-unit-verdict-20261109.json`；端口复现 `experiments/mme-io-audit/mme_io_cpu_probe3.sh`；单位追踪 `experiments/mme-bandwidth-unit/`。速度来自已安装二进制实现，并非获得公开C++ HAL源码。内部BF16枚举12与硬件描述符枚举9不同。


## 视频集成结果（2026-11-09）

隔离匹配A/B：稳态管线100.4545→85.6770秒（−14.71%），DiT80.3465→65.5670秒（−18.39%）。使用POW2Q：Q精确1/16缩放、TPC乘数261.25、shift8 LUT、精确零padding；robust wide。已测0/25/38整层约240.5ms vs323ms；38/head36全部128Q块最大FP32 RMS0.95515%。

**尚不可宣布生产质量通过**：对原生产视频PSNR17.84/18.84dB，独立逐层FP32采样验证已完成：188调用/1788个head-call RMS，最大0.77761%，零个≥1%；仅采样通过，不等于视频质量通过。`h3_validate_video_run.sh`含CPU参考开销，不是速度测试。原生产源码未改，按主HANDOFF最上段查看证据/进程状态。MME-only11.60µs仍未达到10。


## H3整层接入进度（2026-11-09）

`h3_layer_run.sh` + `h3_direct_attention.py` 已验证56头整层：layer0/25/38约240–241ms，同配置生产约323ms。头地址计划缓存、批量Q/K/V暂存、逐quarter转置K、每调用独立输出，robust wide同步。56头各512行FP32抽样最大RMS分别0.334/0.323/0.992%；38层精度裕量很小。跨层25→0→38→25、框架FP8交错、单头回退与输出无别名通过。使用独立160MiB hbm_shared配置。`h3inject/sitecustomize.py`为可选服务注入，视频A/B已完成，结果见顶部。硬件首次故障及恢复前置条件、证据与复现详见主HANDOFF最上段。

## 当前：TPC9.15µs、MME11.60µs、完整head3.927ms

`BUILD_CONTEXT_PROBE=1 bash build.sh` 后运行 `REPEATS=30 bash tenus_runs.sh wide-relaxed real`。layer25/h0 全部输出每次校验，RMS0.376005%。默认 `wide` 保留更强组内同步，约4.127ms、相同SHA。`wide_query=true` 内部1024Q，K/V读HBM，S/P仍留SRAM；使用固定0 15路核、精确零padding、LUT统一缩放2^-8。MME尚未达到10µs；整层与视频实测结果见上。范围检查失败需稳定softmax回退。最新证据、失败G19、构建审计和复现见主HANDOFF最上段。

## 本轮：TPC 约 10 µs，真实 head 4.48 ms（2026-11-09）

`exp_f16_tile_fixed0_g15pad` 配合 `--mme-no-barrier --mme-async-commit --local-qk-barrier`，其余开关延续下述 5.33 ms 配置。真实 layer25/h0 304 块完整校验：4.479 ms/head，RMS 0.376005%。TPC-only 两轮 9.966/10.029 µs/等效128Q块；MME-only 12.185 µs，仍未达 10。新固定0核有范数、溢出、padding误差门限。详细复现、边界和失败实验见主项目 HANDOFF 最新段。

## 最新：5.33 ms/head（2026-11-09 下午）

在下述配置的基础上把 exp 核换成 `exp_f16_tile_affine_skewidx_ms8`（`ablation_runs.sh` 中名为 `xkidx`）。`F16_NOAND` 已证明是错的，不要使用。

## 最新：5.58 ms/head 配置（2026-11-09 下午）

`--tile-rows 512 --group-blocks 16 --scores-fp16 --reload-head-kv --exp-kernel exp_f16_tile_affine_ms8 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa`（`ablation_runs.sh` 中名为 `xgcq`）。新增诊断：`raw_dtype_runs.sh`（FP16/FP32 输出对比）、`raw_np_runs.sh`、`capture_shapes_m.sh` 支持 `ENABLE_PIPELINE_MANAGEMENT`。

## 最新：跨组流水（2026-11-09 中午）

最佳配置：`--tile-rows 512 --group-blocks 16 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --cross-group`，h0 完整 head 6.58 ms。跨组需要 `scal-attention-1.24.1.json` 里 SOB 池为 128（旧配置在 `experiments/`）。剖析用 `--ablation mme-only|tpc-only`、`exp_f16_tile_affine_nomain_ms8`（仅计时）、`raw_bigm_runs.sh`、`raw_nobarrier_runs.sh`。`--prefetch-desc`、`--mme-no-barrier` 结果正确但没有收益。4 行 exp 失败，见 `experiments/exp_f16_rows4_failed.c`。

## 最新：消融、交错 arm、DMA 重叠、FP16 仿射 exp、拆分 fence（2026-11-09 上午）

完整 head（h0，全部块校验）每 head 11.23 → 约 7.7 ms。推荐加 `--interleave-arm --overlap-dma --split-fence`，exp 核用 `exp_f16_affine_ms8`（128 行）或 `exp_f16_tile_affine_ms8`（512 行，`--tile-rows 512 --group-blocks 16`）。`ablation_runs.sh` 可做复测和消融，空 exp 和 `--ablation tpc-only` 只用于计时（`--profile-only`）。256 行没有拆分 fence，也不推荐。8 路 exp 失败，见 `experiments/exp_f16_affine_g8_failed.c`。详情、数据和边界见项目 `HANDOFF.md` 最新段。

## 最新：MME合并与256/512行实验（2026-11-09早间）

`--coalesce-qk` 在固定捕获上验证并合并等价宽度循环；`--tile-rows 256` 用两个key半区，`--tile-rows 512` 用四段key，均保留全部输出。128/256完整head的输出SHA相同，max RMS0.365584%。交叉复测计算约9.62→9.35ms/head（约2.9%改善），含KV/Q/O/stats搬运仍约11ms波动，**没有稳定总耗时或视频加速结论**。512行只做了16块真实验证，没有更快。

```bash
BUILD_CONTEXT_PROBE=1 bash build.sh
python3 check_coalesce_qk.py
bash compare_tiling_runs.sh inputs/head-layer25-h0-g20.bin 20
bash full_head_runs.sh 0 20 256
```

256/512选 `--scores-fp16 --exp-kernel exp_f16_tile_logq_ms8`；512 group须为4的倍数且≤16，尚不支持QK合并。并行PV正确但变慢，默认关闭。失败的G8与padding源码移入 `experiments/`，不在主构建/CLI执行路径中。完整证据、失败原因和后续优先级见项目HANDOFF最新段。

## 最新：完整输出与完整head（2026-11-09）

`runtime_batch.hpp` 已保留每个不同query块的全部输出/统计量，最后统一归一化，再由Synapse DMA回写HPU HBM；`batch_attention_probe.py --group-blocks` 可流式处理完整head。
真实layer25/head0的37750个query全部通过逐块FP32对照，最大RMS **0.3656%**。20块一组，包含每head一次K/V装载、Q搬入、所有O/stats搬回，约 **11.48ms/head**。旧314ms/整层路径尚未被超越；没有视频加速结论。

初版BF16分数在两个query块有1.06%/1.24%误差，CPU复现后改为 **FP16分数、原地BF16概率**（空间相同）。运行真实batch需 `--scores-fp16 --exp-kernel exp_f16_logq_ms8`，未放宽1%门限。

```bash
BUILD_CONTEXT_PROBE=1 bash build.sh
bash full_head_runs.sh 0 20
```

输入按head预整理、K转置、CPU参考、一次性准备与输出scatter不在计时内；仍为隔离同步进程，尚无生产并发/allocator集成。以项目 `HANDOFF.md` 最新段的数值门限、计时范围与后续路线为准。下面保留前阶段微基准记录。

## 最新运行时优化（2026-11-09）

`--compiler-qk --wreg-bulk` 已将保留 master/slave 的编译器 QK 接入双缓冲流水，支持完整 S 的两个带行步长视图和拆分 PV；复用前等待两路 PV。
`--exp-kernel exp_lut_half_logq_b80_ms8 --lut` 用 LUT128 空槽14做 log2 网格修正，三组真实块中位数 **26.3/27.4/26.4 us**，RMS **0.5458%/0.3306%/0.4452%**。旧最佳约31us；默认路径保持不变。

所有数值仍是驻留微基准。`--defer-norm` 只交付最后两块输出，尚未保存每个 query 块的结果；逐块归一化、奇偶不同Q的验证约29.6us/块。100次重放通过环绕和26个框架特殊算子逐位一致检查，没有整层或视频结果。

```bash
BUILD_CONTEXT_PROBE=1 bash build.sh
bash cqk_logq_runs.sh
```

有界 logq 依赖探针的输入上界≤80、两半都有有效 key；离群层与最大值抽样偏差尚未验证。`--compiler-qk` 需对应形状捕获、`--split-pv --pipelined 2..19 --wreg-bulk`。`--alternate-query` 验证不同Q与两份缓冲。完整边界和后续工作以项目 `HANDOFF.md` 最新段为准；下文为早期路径记录。

## 当前已验证

- 固定 128 个 query、head dimension 128；key 数为 128 的倍数，已测到 37,760。
- 一个 MME master；1、2、6、12、24 个 TPC 分摊 query 行。
- 长 key 输出使用 MME 硬件循环；超过 32,768 列时用两个 QK 描述符覆盖。
- BF16 Q/K/V、S/P、分子/输出；FP32 行和；近似 exp 使用范围缩减加五次多项式，不需要系数 LUT。
- 真实输入支持有效 key 长度 37,750：TPC 将 10 个补齐位置的概率置零。
- 检查输出、行和、NaN/Inf、保护区，以及 Q/K/V 和上界是否被改写。
- 超大上界或极小归一化量返回 `UNSUPPORTED`，不能替代离群层的稳定 softmax。

真实 H3 第 25 层的三个 head/query 块，相对 CPU FP32 softmax 的 RMS 误差：

| head | query 起点 | 相对 RMS |
|---:|---:|---:|
| 0 | 0 | 0.00561183 |
| 55 | 10,000 | 0.00349970 |
| 20 | 37,622 | 0.00456988 |

三组都通过保护区、输入未修改和有限数检查。这不是完整视频质量验证。

## 构建与验证

服务器目录：`/home/a/gsv-data/media/mme-direct-20261108/repro/attention`。

```bash
cd /home/a/gsv-data/media/mme-direct-20261108/repro/attention
bash build.sh
bash verify.sh

# 可选：从真实 dump 导出一个 head/query 块，仅使用 CPU。
PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
 /home/a/h3-omni-venv/bin/python export_qkv.py \
 /mnt/sas/tmp/minimax-h3/qkv/qkv-call1-layer25.pt \
 inputs/layer25-h0-q0.bin --head 0 --query-start 0
bash verify.sh inputs/layer25-h0-q0.bin

# 单块提交/等待，30 次计时。
build/attention_runner build/exp_sum.bin build/normalize.bin \
 17 random 37760 24 30 inputs/layer25-h0-q0.bin

# 一次设备提交中连续执行 100 次；五个批次计时，按次数平均。
build/attention_runner build/exp_sum.bin build/normalize.bin \
 17 random 37760 24 100 inputs/layer25-h0-q0.bin batch
```

默认头文件路径 `/opt/habanalabs/src/hl-thunk`，库路径 `/usr/lib/habanalabs`；可用 `HLTHUNK_SOURCE`、`HLTHUNK_LIB` 覆盖。
`BUILD_CONTEXT_PROBE=1 bash build.sh` 另外构建运行时探针。

## 计时口径

计时包含设备提交/等待及四阶段计算；不含初始数据上传、CPU 转置/上界/参考计算、ISA 上传、SO 清零和最终回读。
输入与 K/V 在 SRAM 驻留。这些数字不是一层完整注意力或生视频时间。

- 37,760-key 合成数据，逐块提交：1 TPC 约 2,266 us；24 TPC 约 257 us，存在频率/主机波动。
- 真实 head 0，逐块提交约 300 us。
- 同一真实输入，100 次设备内连续执行、差量更新寄存器：约 157.27 us/块。
- batch 重复使用同一 Q/K/V；它验证设备内重复调度和开销摊销，不代表全部 head/query 的吞吐。

当前仅一个 MME master，尚未证明比已有 H3 注意力实现更快。需要继续扩大 MME 并行、调度重叠，并测实际数据搬运和整层吞吐。
`EXP_FAST_CUBIC` 是实验分支：在真实 head 0 的行和误差约 0.00314，超过当前 0.003 门限；默认构建不启用。

## PyTorch / Synapse 互操作

已验证：

1. 从 `synDeviceGetInfoV2` 取得运行时持有的设备 FD。
2. 用 `habana_frameworks.torch.utils.experimental._data_ptr` 取得真实设备 VA；普通 `data_ptr()` 是框架句柄。
3. SCAL 只读上下文查询后 PyTorch 运算保持正确。
4. Synapse 自有流完成 HPU→SRAM→HPU 拷贝，随后 PyTorch 运算保持正确。
5. 在隔离进程的定制 SCAL 配置中增加专用 `mme_attention0` 流；四次 NOP+完成标记均收到主机完成计数，PyTorch 随后正常。
6. 修正静态命令的 HBM 存储后，ARC ECB 最小 MME 作业在同一上下文执行成功：128×128 BF16 输出 16,384 元素逐位一致，后续框架运算正常。
7. 同上下文 MME、TPC0 ReLU 各连续三次通过，每次后插入框架 GEMM 验证。
8. `runtime_attention_probe.py` 在该上下文完成完整 SRAM 链路，128/1024 keys 合成数据和 37,750 个有效 keys 的真实 H3 输入均通过；真实输入 RMS 0.00561183，保护区和后续框架运算正常。

9. `runtime_pipeline.hpp` 已实现缓存命令/ISA、24 TPC 和设备 fence 的整链一次提交；真实三组输入均通过，逐次输出填毒、框架 GEMM、保护区检查正常。h0 十次均值 238.40 us，h55/h20 各三次均值 240.64/247.05 us，误差与逐阶段路径一致。

10. 2026-11-08 晚：流水路径增加消融计时与可选优化，默认全部关闭（默认路径回归不变）。
    - `--mode floor|lbw|qk|qk-exp|qk-exp-pv|exp|pv|norm`：只跑部分阶段，只有 `full` 检查结果；`stage_summary.py` 由中位数差值得出各阶段耗时。主机端拆分（入队/提交/等待）写在每行 `PIPELINE_REPLAY` 中。
    - `--rearm-only`：monitor 的 config/payload 只在进程内首次写入，此后每块只写 arm（每块 253→53 条 LBW）。
    - `--dual-qk`：QK 按 128 列块切给 `DCORE0_MME_0`/`DCORE2_MME_0`，两者都完成后 SOB 达到 2。
    - `--exp-kernel exp_sum_g2|g4|g4b3|g4b4`：一轮处理 2/4 个独立块（S/P 原地别名，编译器无法自行跨轮重叠）；`b3/b4` 以 2 为底、指数偏置并入 mac 常数，用三次/四次相对 minimax 多项式。
    - `--wreg-bulk`：描述符用 `PACKET_WREG_BULK`；正确，但整链无收益。
    - 真实三组输入在 `--rearm-only --dual-qk --exp-kernel exp_sum_g4b3` 下为 118.3/115.9/115.3 us（基线 237–247 us），RMS 0.005611893/0.003499699/0.004563407。仍是同输入 SRAM 驻留重放。详见 `evidence/stage-opt-summary-20261108.txt` 与项目 `HANDOFF.md`。

11. 2026-11-09：`--lut --exp-kernel exp_lut|exp_lut2`。按 SynapseAI_Core 开源生成器生成 Gaudi2 特殊函数表并上传，TPC 程序开头只写 `LUT_FUNC128_BASE_ADDR`（Synapse 依赖进程内持久的基址；写全部 4 个会改变框架 BF16 sigmoid）。exp 为两遍：精确行最大值，然后 `exp_bf16(s - m)`，即稳定 softmax。真实三组输入 107.4/109.1/110.3 us，RMS 0.5548%/0.3502%/0.4565%；26 个框架特殊算子写入前后逐位一致。`special_ops_check.py`（`run_context_probe.sh special-ops`）可在新进程中把 HPU 特殊算子与 CPU 对照。

未验证：生产并发/allocator/调度生命周期，以及整层/视频性能。
不加 `--pipeline` 的诊断路径仍逐阶段 CPU 等待，真实长 key 约 5.5 ms/块。缓存路径的计时含 monitor 配置和单次提交等待，但不含初始数据 DMA 和一次性准备；是同输入 SRAM 驻留微基准。当前仅一个 MME master，不能据此推断视频加速。

```bash
BUILD_CONTEXT_PROBE=1 bash build.sh
ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-attention \
  --input inputs/layer25-h0-q0.bin --pipeline --tpcs 24 --repeats 10
```

此计划只支持固定形状/地址、隔离且框架已同步的单进程；64 KiB 命令缓冲不环绕，拒绝空间不足，分配保留至进程退出。不要并发调用。私有 SOB/monitor 已通过配置预留，不挪用框架资源。
裸驱动队列混用、SCAL 直接 LBW 写 MME 描述符、首版 host_shared 静态命令 ECB 提交都失败。修正后的最小 MME 探针仍不能直接作为 vLLM 实现。

最新 ECB 失败：QMAN 在 HBM 地址读到全零指令，而程序实际放在 `host_shared`；转储显示源地址高 32 位使用 HBM 高位。
修复版静态命令存放 HBM，并回读验证，现已通过 MME、TPC 与完整长注意力；设备内同步与 24 TPC 并行也已通过；下一步提升 MME 利用率、验证整层吞吐。详见项目 `HANDOFF.md`。

本轮错误计数曾为 3；ECB 在两个 MME QMAN 上各触发一次 undefined opcode，计数变为 5。设备已恢复空闲；未清零计数。

## 文件与来源

- `attention_runner.cpp`：独立 SRAM 链路、CPU 参考、设备内批量重复、差量寄存器更新。
- `kernels/`：TPC 源码；`reduce_no_lut.h` 避免内置 reduce 隐式发出 LUT lookup。
- `context_bridge.cpp`、`context_probe.py`：运行时研究探针，包含尚未验证的实验函数。
- `runtime_pipeline.hpp`、`runtime_attention_probe.py`：同上下文缓存注意力与设备同步，按上述命令单独验证。
- `scal-attention-1.24.1.json`：从当前 SCAL 默认配置导出后增加私有测试流与主机完成计数器，仅用于隔离实验进程。
- `fixtures/`、`evidence/`：描述符导出、验证与失败日志。
- `third_party/synapseai-core`：必要寄存器头文件，来自 [SynapseAI_Core](https://github.com/HabanaAI/SynapseAI_Core)，commit `d4d7f5c6f9f9cfc34d2a97b3773901d15443d4ee`，附原许可证。
- `third_party/hcl`：SCAL 及 ARC 包头，来自 [HCL](https://github.com/HabanaAI/HCL)，commit `11f9114b88038174d8b2c9ac09e2295b75455078`；上游 README 声明 Apache-2.0，ARC 头自带 MIT 标识。它们不是当前 SCAL 1.24.1 的版本匹配证明。

完整 DFA 转储仍在服务器 `evidence/context-hl/dfa_log.txt`，另有本地压缩副本。
