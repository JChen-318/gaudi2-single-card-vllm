# 配置说明

## config/serve.env

| 项 | 默认 | 说明 |
|---|---|---|
| `TARGET_MODEL` / `DRAFT_MODEL` | — | 目标模型与 DFlash2 草稿模型的目录（必填） |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-fp8-mtp` | 请求里 `model` 字段用的名字 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 监听地址。`0.0.0.0` 会对外开放，请配合 `API_KEY` 和防火墙 |
| `API_KEY` | 空 | 设置后 `/v1/*` 需要 `Authorization: Bearer <key>`；`scripts/smoke_test.sh`、`bench.sh` 会自动带上 |
| `MAX_MODEL_LEN` | 262144 | 单个请求的最大上下文（提示词 + 生成） |
| `MAX_NUM_SEQS` | 2 | 同时处理的请求数（1–16）。只测过 1–2 |
| `KV_CACHE_GIB` | 29 | KV 缓存显存（FP8），29 GiB 约 52.8 万 token（2048 token 的块；不开前缀缓存时块是 1664 token，28 GiB 就有 52.7 万） |
| `PREFIX_CACHE` | 1 | 前缀缓存：请求开头与之前某个请求相同的部分直接复用（第 18 节）。0 = 关闭 |
| `PREFIX_STATE_SLOTS` | 16 | GDN 状态快照个数，每个约 155 MB 显存，按最近使用淘汰 |
| `PREFIX_STATE_INTERVAL` | 16384 | 除每个提示词末尾前的最后一个 2048 边界外，每隔多少 token 存一份快照（2048 的倍数） |
| `PIN_MAIN_CPU` | 空 | 预热后把引擎主线程固定到这个 CPU，并让其他线程避开它和它的超线程兄弟。选一个空闲的物理核，约省 0.3 ms/轮 |
| `DATA_DIR` | `~/.cache/gaudi2-single-card-vllm` | 编译缓存、日志、运行期控制文件 |
| `GENERATION_CONFIG` | `temperature 1.0, top_p 0.95, top_k 20` | 请求没给采样参数时的默认值 |
| `STEP_LOG` | 空 | 设为文件路径后，每轮 decode 记一行主机计时（`tools/trace/steps_stats.py` 可读）。长期运行会持续变大，只在测速时打开 |

`scripts/serve.sh --skip-warmup` 跳过预热（约 6 分钟可用，只用于功能检查）。

## 改并发数或上下文长度

decode 分桶由 `serve.sh` 根据配置计算：batch 取 1 到 `MAX_NUM_SEQS`，页数上限取「所有请求都满长度」时的页数（每页 128 token），但不超过 KV 缓存能放下的页数。实测配置（262144 × 2，29 GiB）得到 4096 页。

- 分桶越多，预热越久。`MAX_NUM_SEQS=1` 时页数减半，预热也更短。
- `MAX_NUM_SEQS` 大于 2 时，融合 GDN 核心只覆盖 batch ≤ 2（`FLASHINFER_GAUDI_MTP_PREPARED_MAX_BATCH=2`），更大的 batch 会走 PyTorch 参考实现，结果正确但较慢。这一组合没有测过速度。
- 加大 `KV_CACHE_GIB` 前先看启动日志里的显存占用：权重约 32.5 GB，预热编译的图约 12.8 GiB，前缀缓存的快照约 2.4 GiB（16 个），卡上共 94.6 GiB。

## serve.sh 中的调优开关

这些开关都写在 `scripts/serve.sh` 里，就是实测用的配置。大部分在代码中默认关闭，只有显式打开才生效。章节号对应 [优化说明](OPTIMIZATION.md)。

| 开关 | 作用 | 章节 |
|---|---|---|
| `VLLM_HPU_SHARED_PREFIX_DECODE`、`VLLM_HPU_SKIP_UNBACKED_BLOCK_LIST`、`VLLM_HPU_SHARE_GROUP_MASKS` | 验证 token 共用上下文 KV，长上下文不再按 token 重复读取 | 2 |
| `VLLM_HPU_FAST_DECODE_PREP`（`_VERIFY`）、`VLLM_HPU_ENVS_CACHE` | 向量化的 decode 输入准备，前 256 轮与旧实现逐元素比对 | 3 |
| `VLLM_HPU_UNIQUE_COMPILE_CODE`、`VLLM_HPU_FUSED_MODEL_ENTRY`、`VLLM_HPU_DFLASH2_DRAFT_STACK` | 编译区域各用独立代码对象、第一组内嵌嵌入层、草稿模型整体编译 | 9 |
| `VLLM_HPU_DFLASH2_LOOKAHEAD`（`_SAMPLE`、`_DELTA`） | 按预测提前提交下一轮主模型和采样，下一轮输入增量准备 | 9 |
| `VLLM_HPU_DFLASH2_PAGE_BUCKETS`、`_DIRECT_CHECKPOINTS` | 草稿上下文页数补齐到桶；GDN 8 个检查点直接写入状态池 | 9 |
| `VLLM_HPU_DFLASH2_CPU_INPUTS`、`_CONTEXT_REGION`（`decode-control.json`） | 接受长度和草稿输入留在设备上准备，草稿可以在采样结果回到主机前排队；目标模型特征的选取与写入编译成一个区域 | — |
| `VLLM_HPU_DFLASH2_HOST_OPT`（`host-control.json`） | 复用选出的采样标量，主机侧的小草稿输入打包传输 | — |
| `VLLM_HPU_FAST_TOPK_TOPP`、`VLLM_HPU_DFLASH2_CHUNKED_TOPK`、`VLLM_HPU_LM_HEAD_FP8` | 采样与草稿候选不对整个词表排序；词表投影用 FP8 | 9 |
| `VLLM_HPU_DFLASH2_PREDRAFT`（`_VERIFY`） | 草稿不等主机读回，紧跟采样在设备上提交 | 11 |
| `VLLM_HPU_GUARD_SKIP`、`VLLM_HPU_RNG_POOL` | 稳定轮次跳过 Dynamo 守卫检查；拒绝采样的随机数批量生成 | 11 |
| `VLLM_HPU_DFLASH2_EARLY_READBACK` | 采样结果在流内拷到固定内存，提前读回 | 12 |
| `VLLM_HPU_PIN_MAIN_CPU` | 引擎线程绑核（由 `PIN_MAIN_CPU` 设置） | 12 |
| `FLASHINFER_GAUDI_MTP_FUSED`、`FLASHINFER_GAUDI_MTP_FUSED_GATHER` | 融合 GDN 验证核心；先取初始状态，避免每层多一次检查点复制 | 13、15 |
| `VLLM_HPU_SHARED_PREFIX_ATTN=dense`（`VLLM_HPU_DENSE_ATTN_QK` / `_AV`） | 全注意力对每个请求的整段上下文做一次 softmax，KV 只 gather 一次；QK/AV 用 FP8 | 17 |
| `--enable-prefix-caching` + `VLLM_HPU_PREFIX_STATE_CACHE`（`_SLOTS`、`_INTERVAL`、`_BLOCK`） | 前缀缓存：注意力 KV（目标与草稿）走 vLLM 的前缀缓存，GDN 状态在固定位置存设备快照，命中时恢复到请求的紧凑状态槽；调度块 2048 token。由 `PREFIX_CACHE` 控制 | 18 |
| `VLLM_HPU_PROMPT_FINE_CTX_STEP` / `_MIN` | 2048 token 的 prefill 段在上下文 256 页以上每 64 页（8K token）加一个分桶，补齐不超过 8K | 18 |
| `VLLM_HPU_PREFILL_SPLIT_ATTN` | 上述分桶上的 prefill 段，对已缓存上下文的注意力切成 8K 一片、叠成一个批次一次 FusedSDPA；只有最后一片带掩码，图与上下文长度无关 | 18 |
| `--kv-cache-dtype fp8_inc`（同时 `env -u QUANT_CONFIG`） | FP8 KV 缓存，不校准 | 14 |
| `VLLM_HPU_WARMUP_PHASE_GC` | 预热时不在每个分桶前后强制 `gc.collect()` | 14 |
| `TORCH_COMPILE_JOB_ID` + `TORCHINDUCTOR_CACHE_DIR` | Dynamo 本地 PGO：记住哪些维度是动态的，下次启动直接按动态编译 | 15 |
| `GC_KERNEL_PATH` + `KIRO_TPC_LIBS` + `PT_HPU_RECIPE_CACHE_CONFIG` | 三个内核库经转接层加载；设备端编译结果缓存到磁盘 | 9.4 |

运行期控制文件由 `serve.sh` 在 `$DATA_DIR/run/` 下生成。`control.json` 里的各项可以在服务运行中修改，批次变化时重新读取，便于同进程对照测试：

```json
{"lookahead": true, "fast_topk": true, "presample": true, "delta": true, "predraft": true,
 "guard_skip": true, "rng_pool": true, "early_readback": true}
```

## 诊断

| 需要 | 做法 |
|---|---|
| 每轮耗时 | `config/serve.env` 里设 `STEP_LOG`，重启后用 `tools/trace/steps_stats.py` 读 |
| 预热时编译时间花在哪 | 在 `serve.sh` 的环境变量里加 `VLLM_HPU_COMPILE_TIMES=1`，预热结束后日志里会有分解 |
| 为什么重新编译 | 加 `TORCH_LOGS=recompiles` |
| 设备 trace | 见 [tools/README.md](../tools/README.md)：需要 profiler 配置，一次只采一两轮；导出时主机内存占用很大 |

## 以 systemd 服务运行

```ini
# /etc/systemd/system/gaudi2-vllm.service
[Unit]
Description=Gaudi2 single-card vLLM
After=network-online.target

[Service]
User=<运行用户>
WorkingDirectory=/path/to/gaudi2-single-card-vllm
ExecStart=/path/to/gaudi2-single-card-vllm/scripts/serve.sh
KillSignal=SIGTERM
TimeoutStopSec=120
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

`serve.sh` 不加 `--daemon` 时在前台运行，适合交给 systemd 管理。`journalctl -u gaudi2-vllm -f` 查看日志。
