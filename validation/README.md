# 离线验证

这里的测试不需要 Gaudi 设备，也不需要安装 Habana 软件栈，只要 `torch`、`numpy` 和 `pytest`。

被测代码按 AST 从本仓库的源码文件里抽出来执行（`vllm-gaudi/vllm_gaudi/extension/ops.py`、`.../v1/worker/hpu_model_runner.py` 等），不是另写的副本，所以测的就是发布的那份文本。

```bash
python -m pytest -q validation        # 在仓库根目录执行，约 3 分钟，2588 项
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
