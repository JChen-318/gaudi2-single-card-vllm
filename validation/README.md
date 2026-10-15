# 离线验证

这里的测试不需要 Gaudi 设备，也不需要安装 Habana 软件栈，只要 `torch`、`numpy` 和 `pytest`。

被测代码按 AST 从本仓库的源码文件里抽出来执行（`vllm-gaudi/vllm_gaudi/extension/ops.py`、`.../v1/worker/hpu_model_runner.py` 等），不是另写的副本，所以测的就是发布的那份文本。

```bash
python -m pytest -q validation        # 在仓库根目录执行，约 3.5 分钟，2633 项
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
| `test_v13_warmup_gc.py` | 预热按阶段 gc（`VLLM_HPU_WARMUP_PHASE_GC`）：每个分桶只读显存增量、不调用 `gc.collect()`，出错时也能退出；runner 中开关与阶段前后各一次 gc 的接入，默认关闭 |

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
