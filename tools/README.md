# 测速与 trace 分析工具

这些脚本只依赖 Python 标准库，在服务器本机对 `127.0.0.1:8000` 的 OpenAI 兼容接口发请求。模型名写的是本地部署用的 `qwen3.8-27b-fp8-mtp`，按需修改。

## 测速（`bench/`）

| 脚本 | 用途 |
|---|---|
| `bench.py <tag> [max_tokens] [prompt_id]` | 代码/技术类短提示词，流式；输出首 token 时间和 decode 速度 |
| `bench_ctx.py <tag> <近似 token 数> [max_tokens]` | 日志类长提示词；额外输出提示词 token 数、每轮耗时、每轮接受数 |
| `greedy2.py <out.json> [max_tokens] [sizes]` | `temperature=0` 输出集合：短提示词、长上下文、2 路并发 |
| `gcmp.py <a.json> <b.json>` | 比较两份 `greedy2.py` 输出，给出公共前缀长度 |
| `bench_multi.py [max_tokens]` | 15 类提示词（中英文解释与写作、多种代码、数学、翻译、摘要、JSON、表格、思考模式、复述）逐条单路测速，再测 3 组两路并发；输出每轮接受草稿数和每轮耗时 |

说明：

- `bench_ctx.py` 的每轮耗时来自 `/metrics` 里草稿计数的前后差，只在单客户端时准确。两个请求并发时，把它报的 `ms_per_round` 乘 2 才是真实的每轮时间。
- 提示词实际 token 数大约是「近似 token 数」参数的 2 倍。
- 速度取决于每轮接受数，受采样随机性影响。比较不同版本时优先看每轮耗时。
- 单路 `temperature=0` 的输出在这个部署上不可逐字复现，`gcmp.py` 的结果只对 2 路并发有判定意义。

## trace 分析（`trace/`）

输入是 vLLM torch profiler 导出的 `*.pt.trace.json.gz`。

| 脚本 | 用途 |
|---|---|
| `trace_hw.py <trace>` | 一轮内的设备忙碌、空闲间隙、各阶段的内核耗时 |
| `trace_hw_cmp.py <label=trace> ...` | 多份 trace 的内核耗时并排比较，按首尾差值排序 |
| `trace_idle.py <trace>` | 设备空闲按主机阶段和间隙大小拆分 |
| `gb_parse.py <log>` | 汇总 `TORCH_LOGS=graph_breaks` 的输出 |
| `trace_regions.py <trace>` | 逐个编译区域列出主机提交时间、缓存查找耗时、自上一区域以来的设备空闲，以及主机领先设备多少 |
| `steps_stats.py <steps.log>` | 读 `VLLM_HPU_STEP_TIMER` 日志，按并发数给出每轮耗时的中位数和分位数（无需 profiler） |
| `phase_stats.py <steps.log>` | 同一日志里的阶段标记：读回、下一轮准备、草稿提交、预提交主模型、预提交采样之间的主机耗时 |

### 采集方法

启动时加 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>","torch_profiler_use_gzip":true}'` 和 `VLLM_PROFILER_ENABLED=full`，然后在请求生成中途连续调用 `POST /start_profile` 和 `POST /stop_profile`，只采一轮。

### 注意事项

- **硬件 trace 默认是开的。** 采集整个请求会让 EngineCore 内存涨到 30 GiB 以上，`stop_profile` 可能卡住半小时。要么只采一两轮，要么用 `hl-prof-config --hw-trace off` 生成配置并通过 `HABANA_PROF_CONFIG` 指定。
- `hl-prof-config` 不接标准输入时会挂住，调用时加 `< /dev/null`。
- 开 profiler 后的第一次采集带启动开销，主机侧耗时偏大，不要拿它做前后对比；设备侧内核耗时不受影响。
- 只设 `VLLM_TORCH_PROFILER_DIR` 不会注册 profile 路由，必须用 `--profiler-config`。
- 带 Python 栈的 profiler 会放大主机侧 Python 的耗时，绝对值不可信，只看比例。
- 诊断配置不是性能配置，正式测速要去掉 profiler 和断图日志。

## 其它

`bucket_count.py` 复算线性分桶策略在不同 `VLLM_DECODE_BLOCK_BUCKET_MAX` 下会生成多少个 decode 桶。

`dynamo_lookup_bench.py [N] [层数]` 在 CPU 上测量 N 个同类模块共用一个 forward 代码对象时的 Dynamo 缓存查找耗时，并与独立代码对象对照（说明 `VLLM_HPU_UNIQUE_COMPILE_CODE` 的来由）。

**测速口径**：做同进程 A/B 时，优先用 `VLLM_HPU_STEP_TIMER` 加 `steps_stats.py` 的每轮中位数，它能分辨 0.1 ms 级的差别；客户端的「首尾时间差 / 轮数」受首尾轮和接受数波动影响，噪声约 ±1 ms。
