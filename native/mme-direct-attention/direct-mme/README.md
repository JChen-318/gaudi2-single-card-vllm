# Gaudi2 直接 MME 编程：已验证的最小实现

2026-11-08，在 HL-225 / module 7 / PCI `0000:08:00.0` 上验证。
服务器：`/home/a/gsv-data/media/mme-direct-20261108/repro/`。

本轮已实现 **不经图编译器执行的 128×128 BF16 GEMM**，以及 **中间结果保留在 SRAM 的两次 GEMM 串联**。
运行程序只链接 `libhl-thunk-tests.so`、`libhl-thunk.so` 和系统库，不加载 Synapse / SCAL / PyTorch。
描述符模板最初由 SynapseAI 1.24.1 生成，因此目前是固定形状的描述符复用实现，还不是独立的通用描述符生成器。

## 运行

在服务器、卡空闲时：

```bash
cd /home/a/gsv-data/media/mme-direct-20261108/repro
bash build.sh
bash verify.sh

# 单独运行
build/mme_runner --mode hbm --seed 17
build/mme_runner --mode sram --seed 17
build/mme_runner --mode chain --seed 17
```

`build.sh` 默认使用 `/opt/habanalabs/src/hl-thunk` 的头文件和 `/usr/lib/habanalabs` 的库；
可用 `HLTHUNK_SOURCE`、`HLTHUNK_LIB` 指定其他位置，但其他版本尚未验证。
需要 GCC、安装好的 hl-thunk 源码/库和 `/usr/include/drm/drm.h`。
不需要 sudo，不需要启动 Docker，不修改驱动或全局配置。

`mme_runner` 可用 `--busid` 改 PCI 地址；两个 smoke 程序目前固定使用上述服务器地址。
设备获取为独占方式，运行时不能有另一个 HPU 作业。

## 已通过的验证

| 模式 | 数据与操作 | 对比 |
|---|---|---|
| `hbm` | A、B、C 在 HBM；C = BF16(A×B) | 全部 16,384 个输出 |
| `sram` | A、B、C 在 SRAM；C = BF16(A×B) | 全部 16,384 个输出 |
| `chain` | A、B、C、D 在 SRAM；C = BF16(A×B)，D = BF16(C×B) | 两份输出共 32,768 个元素 |

每个模式执行三组随机整数输入（seed 1 / 17 / 20261108）和一组 B=I 的输入（seed 42）。
共 12 个用例、262,144 个 BF16 元素，`bad=0`、`max_abs_error=0`、`guard_bad=0`。
随机输入范围为 -4 到 4；CPU 参考用 FP32 累加，每次 GEMM 输出按 round-to-nearest-even 转 BF16。
这些输入可用于严格逐位校验，不代表已验证所有浮点分布、舍入边界、NaN/Inf 或其他矩阵尺寸。

链式测试在两次 GEMM 全部完成后才读回 C 和 D，C 在两次计算之间没有 DMA 到 HBM 或主机。
所有输出之后均有 4 KB 的 `0xcd` 保护区并检查其完整性。
另外验证了独立的 4 KB HBM DMA 往返，以及 MME 队列的 CP_DMA → SOB → monitor/fence 链路。
完整记录见 `evidence/verify.log`；环境与库哈希见 `evidence/environment.txt`。

## 执行链

1. `hltests_init` / `hltests_open`，查询设备 SRAM 地址、可用 SOB / monitor 和队列。
2. 用 DMA 将输入写入指定 HBM 或 SRAM 地址。
3. 在主机映射内存构造 common-CP 命令：写描述符寄存器，再写 `MME_CTRL_LO_CMD`。
4. 在 SRAM `base+0x2000` 放置 upper-CP 的 CP_DMA 包，使 MME 的 common CP 读取上述命令。
5. MME 完成后向指定 SOB 加 1；DMA-up 队列的 monitor/fence 等待此值，再读回输出。
6. `chain` 在同一个 MME common-CP 中等待第一次 GEMM 的 SOB，再写第二个描述符并触发计算。

数据区从 `sram_base+0x10000` 开始。测试由设备独占保证这块 SRAM 不与其他工作负载冲突。
SRAM 测试没有分配 HBM 数据缓冲区。

## 描述符与启动要点

| 项目 | 当前设置 | 依据 |
|---|---|---|
| MME 队列 | `hltests_get_mme_qid(fd, 0, STREAM0)`，本机为 16 | 安装的 hl-thunk Gaudi2 源码与上卡验证 |
| 描述符寄存器 | QMAN 相对地址 `0xb008` 至 `0xb2dc`，182 个 DWORD | 驱动寄存器头文件、编译器输出 |
| 触发寄存器 | `0xb004 = 0x3ff` | 驱动 CMD 位定义与上卡验证 |
| `RATE_LIMITER` | `0xb29c = 0x040404` | 正常 GEMM 后的寄存器读回，见 JSON 证据 |
| 单 master | 关闭 brains 的 slave enable 和 master/slave 同步握手 | 固定 GEMM128 模板的多 seed 上卡验证 |
| 完成信号 | `so0Addr` 指向可用 SOB，`so0Val = 0x80000001` | 驱动字段定义与上卡验证 |
| 第二次计算等待 | common-CP monitor/fence，`cmdq_fence=true` | hl-thunk 参考用法与链式验证 |

最关键的发现：**文本 MME 描述符没有打印 RATE_LIMITER**。
将未打印字段全部补零会让输入 AGU 不前进，GEMM 无法完成。
补上正常运行观察到的 `0x040404` 后，同一条底层路径立即通过计算校验。
仅依据旧 `decode_mme.py` 的输出也无法构造完整命令：QMAN 日志中的 WREG_BULK 最多只打印前 20 项。

`generate_desc.py` 从 `fixtures/gemm128-desc0.txt` 和 `evidence/runtime-registers.json` 重建固定模板。
它会关闭当前小矩阵不需要的 slave 执行/握手，并清零性能事件。
偏置、USER_DATA、PCU、padding 等按本 fixture 的零值处理。

```bash
python3 generate_desc.py fixtures/gemm128-desc0.txt \
  --runtime-registers evidence/runtime-registers.json --output build/regenerated.h
cmp desc128.h build/regenerated.h
```

`mme_capture.py` 的原始参考描述符来自项目 `scripts/` 中的基线脚本。
重新采集需要 v17 镜像和该服务器的 PyTorch venv；运行最终 C 程序不需要它们。

## 失败实验与设备状态

在确定 RATE_LIMITER 之前，两次直接 GEMM 没有完成，进程退出后驱动因 outstanding CS 自动重置设备。
一次正常 GEMM 后的调试采集误读了受保护的 `0x1000007ffc0cb410` 等配置寄存器，产生
`PCIE_ADDR_DEC_ERR` / `MME0_QMAN_SW_ERROR` 并触发自动重置。
最终交付的构建和验证程序不执行 debugfs 寄存器扫描；JSON 证据仅保留描述符范围及状态/命令字段。
不要将该 JSON 的偏移列表作为扫描其他配置块的依据。

最终验证后设备为 `Operational`，温度 31°C、768 MiB 已用、利用率 0%，无计算进程。
`hl-smi` 的 `Volatile Uncor-Events` 为 2；会话开始时为 0。
上述 PCIe/配置访问错误的内核日志另存 `evidence/kernel-research-events.log`，没有清零计数器。

## 尚未完成

- 形状固定为 M=N=K=128，尚未独立生成任意尺寸/转置/分块的描述符。
- 仅验证一个 MME master；没有多 MME 并行和完整 master/slave 调度。
- `chain` 是 MME → MME，尚未加入 TPC 的 exp、行和或归一化。
- 尚未实现 H3 的 QKᵀ → exp/行和 → PV，更没有端到端提速或画面质量结论。
- 没有进行可靠性能测量；当前结论是计算与 SRAM 通路的可行性。

下一步应先验证 SRAM 上独立 TPC kernel 的加载、启动与完成信号，再接成
MME → TPC → MME，并用不同矩阵尺寸和真实 Q/K/V 检查布局、数值及尾块。

## 源码参考

使用安装的 1.24.1-482 版 hl-thunk 和驱动头文件为实际执行依据。
开源接口与参考结构可查 [hl-thunk](https://github.com/HabanaAI/hl-thunk-open)、
[Gaudi1 MME 测试](https://github.com/HabanaAI/hl-thunk-open/blob/2a0673214b7ac03fefac5639b53d63fc3206ec49/tests/gaudi/gaudi_mme_conv.c) 和
[SynapseAI Core](https://github.com/HabanaAI/SynapseAI_Core)。
Gaudi1 测试只用于理解提交与同步结构，不能直接套用其寄存器或描述符到 Gaudi2。
