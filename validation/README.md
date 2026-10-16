# 离线验证

这里的测试不需要 Gaudi 设备，也不需要安装 Habana 软件栈，只要 `torch`、`numpy` 和 `pytest`。

被测代码按 AST 从本仓库的源码文件里抽出来执行（`vllm-gaudi/vllm_gaudi/extension/ops.py`、`.../v1/worker/hpu_model_runner.py` 等），不是另写的副本，所以测的就是发布的那份文本。

```bash
python -m pytest -q validation        # 在仓库根目录执行，约 3 分钟
```

| 文件 | 覆盖内容 |
|---|---|
| `test_shared_prefix.py` | 共享前缀验证（`VLLM_HPU_SHARED_PREFIX_DECODE`）：新注意力与展平布局数值一致（float64 到 1e-9）、独立的朴素 softmax 对照、跨页时不可见页不影响结果、bf16/float32、真实头数形状、主机侧页表构建、开关关闭时行为不变 |
| `test_group_mask_sharing.py` | 注意力组共用掩码（`VLLM_HPU_SHARE_GROUP_MASKS`）：共用与不共用的元数据和注意力输出完全相同 |
| `test_decode_fast_paths.py` | 向量化的 decode 输入准备（`VLLM_HPU_FAST_DECODE_PREP`）与列表实现逐元素相等 |
| `test_v6_compile.py` | 独立代码对象保持输出不变并分开 Dynamo 缓存；第一层组内嵌嵌入层；草稿模型整体编译与原 forward 一致（CPU `torch.compile`） |
| `test_fast_topk.py` | 两级 top-k 与 `torch.topk` 值一致；免排序 top-k/top-p 与 vllm-core 的 `apply_top_k_top_p_pytorch` 保留集合一致（含同分边界策略）；候选 top-16 与 `top_k_reference` 一致 |
| `test_lookahead.py` | 下一轮预提交：完全复用、一路结束后只取另一路的行、请求结束丢弃、预测不符时关闭、接近 max_tokens 不启用、先准备后提交（假 runner） |
| `test_delta_prepare.py` | 增量准备与完整准备逐字段一致（120 轮、单/两路、跨页、M-RoPE），页表变化时退回完整准备，比对失败时关闭 |
| `test_draft_page_buckets.py` | 草稿上下文页数分桶：原有页上的输入逐元素相同，补出的页是 pad 块且偏置全为 −inf |
| `test_v11_predraft.py` | 草稿提前：主机侧输入；设备桥接与主机草稿路径在所有接受长度、有无滑窗、单/两路下看到的上下文 slot、查询 slot、输入和 KV 写入位置一致（需要 FlashInfer-Gaudi 的 `dflash2.py`，用 `FLASHINFER_DFLASH2_PY` 指定，否则跳过）；随提前提交保存、一路结束时取对应行、有 prefill 时不用、校验失败时关闭；提前读回（`VLLM_HPU_DFLASH2_EARLY_READBACK`）在完全复用时使用并逐次比对、部分复用时跳过、不一致时关闭、默认关闭 |
| `test_v11_guard_skip.py` | 守卫跳过：预热/跳过/复查节奏、产生新图时重新计数、未见过的输入在跳过状态下用完整守卫重试且结果正确、嵌套编译区域不断图、轮次键只看形状和标志（CPU `aot_eager`） |
| `test_v11_rng_pool.py` | 采样随机数池：每轮取到独立的拷贝、按轮数补充、分布检查、有独立生成器的请求不走池、拒绝采样器的接入 |
| `test_v11_thread_isolation.py` | 引擎线程隔离（`VLLM_HPU_PIN_MAIN_CPU`）：主线程固定到指定核，其余线程全部避开该核及其超线程兄弟，之后新建的线程在刷新时挪开；核不可用时不做任何绑定 |
| `test_v11_compile_timers.py` | 预热编译时间分解（`VLLM_HPU_COMPILE_TIMES`）：包住 Habana 后端各 pass 和 Synapse 图编译并在日志中汇总；后端不存在时不启用（用替身模块） |
| `test_v13_fused_gather.py` | 融合 GDN 核心先取初始状态（`FLASHINFER_GAUDI_MTP_FUSED_GATHER`）：用与 kernel 约定相同的 Python 替身，两条路线的输出与状态池逐位相同（单/两路、接受数含 0 与超过 8、直接写检查点与 index_copy 两种写法）；kernel 只拿到选中的行，槽位指向本行；默认开启 |
| `test_dense_decode_attn.py` | 整段计算的共享前缀注意力（`VLLM_HPU_SHARED_PREFIX_ATTN=dense`）：用生产的页表、归属与可见性生成方式，FP8/BF16 缓存、4 种矩阵乘精度组合与 float64 因果注意力参考一致；补齐的请求行输出为 0 且无 NaN；改动另一个请求的页不影响本请求 |
| `test_v13_warmup_gc.py` | 预热按阶段 gc（`VLLM_HPU_WARMUP_PHASE_GC`）：每个分桶只读显存增量、不调用 `gc.collect()`，出错时也能退出；runner 中开关与阶段前后各一次 gc 的接入，默认关闭 |

`core/` 下的测试用到 vLLM 本体（本仓库的 `vllm-core/` 或已安装的 vLLM）和它的依赖，需要在装好服务的环境里跑；不加载平台插件、不碰设备：

```bash
VLLM_PLUGINS=none TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=vllm-gaudi .venv/bin/python -m pytest -q validation/core
```

| 文件 | 覆盖内容 |
|---|---|
| `core/test_prefix_state_cache.py` | 前缀缓存（`VLLM_HPU_PREFIX_STATE_CACHE`）：在 vLLM 真实的 KV 缓存管理器上（目标、草稿两个注意力组 + 两个 GDN 组，align 模式，每请求 7 个投机块），多轮续写命中上一轮提示词末尾的快照，中途分叉时退到更早的快照，快照被挤掉后不命中，无快照时即使注意力 KV 还在也不命中；投机 decode 之后下一轮仍能命中；DFlash 的 EAGLE 标记被清除；被抢占的请求恢复时也恢复状态；同一步里恢复和保存不会用到同一个槽位；分段在快照位置停下；runner 侧只复制每组的基础状态行 |
| `core/test_prefill_buckets_slicing.py` | 细分上下文分桶（`VLLM_HPU_PROMPT_FINE_CTX_STEP`）只加给最大的 query、查找取到最近的桶、冻结的每 query 上下文索引；分片 prefill 注意力（`VLLM_HPU_PREFILL_SPLIT_ATTN`）用 CPU 替身 kernel 与稠密 softmax 注意力一致（GQA、补齐页填大值、归一化常数因子可抵消），以及适用条件 |

`dynamo/` 下是两个断图的复现与检查脚本，需要与服务器同主版本的 torch（这里用的是 2.11）：

```bash
python validation/dynamo/check_selector_graph.py        # 期望：graphs: 1  graph breaks: 0
python validation/dynamo/repro_fsdpa_unbacked.py mark   # 复现：出现第二种 batch 大小后断图
python validation/dynamo/repro_fsdpa_unbacked.py nomark # 去掉标记后不断图
```

## 局限

- 这些测试证明的是主机侧数据结构和注意力数学在 CPU 上等价，不能代替真机验证。真机上抓到过一个这里原先没覆盖的问题（返回张量内存不连续），之后补了断言。
- 没有覆盖 GDN 层和调度；下一轮预提交、增量准备用的是假 runner，真机上另有逐字段比对和不一致计数（见优化说明第 6 节）。
- 源码目录可用环境变量 `VLLM_GAUDI_SRC` 指向别处（`vllm_gaudi` 包目录）。
