# 单文件 TPC 内核库转接层

Gaudi 的图编译器从 `GC_KERNEL_PATH` 加载 TPC 内核库。本项目同时用到三个库：自定义的随机采样内核、FlashInfer-Gaudi 内核和官方 `libtpc_kernels.so`，原来以冒号分隔写在 `GC_KERNEL_PATH` 里。

打开编译结果磁盘缓存（`PT_HPU_RECIPE_CACHE_CONFIG`）后，运行时要求 `GC_KERNEL_PATH` 是**单个普通文件**，否则引擎初始化直接失败：

```text
FATAL ERROR :: MODULE:PT_SYNHELPER Flag GC_KERNEL_PATH: Path is not a regular file: a.so:b.so:c.so
```

这个转接库把多个库合成一个入口：

- 按 `KIRO_TPC_LIBS`（冒号分隔，顺序与原来的 `GC_KERNEL_PATH` 相同）加载各库，合并它们报告的 GUID；同名 GUID 以靠前的库为准。
- `InstantiateTpcKernel`、`GetShapeInference`、`GetSupportedDataLayouts`、`GetSuggestedManipulation` 按 GUID 转给报告它的库。
- 官方库作为链接依赖（DT_NEEDED），它的旧接口（`HabanaKernel`、`GetKernelNames` 等）通过 `dlsym` 仍能找到；`GetLibVersion` 返回官方库的版本。
- 自定义库没有 `GetSupportedDataLayouts` 时，对其内核报告一个与布局无关的 `xxxxx` 布局。
- 未设置 `KIRO_TPC_LIBS` 时只加载官方库。`KIRO_TPC_SHIM_LOG=1` 打印路由日志。

## 构建与检查

```bash
g++ -O2 -fPIC -shared -std=c++17 -I/usr/include/habanalabs tpc_kernel_shim.cpp -o libkiro_tpc_shim.so \
  -Wl,--no-as-needed -L/usr/lib/habanalabs -ltpc_kernels -ldl -lpthread -Wl,-rpath,/usr/lib/habanalabs
g++ -O1 -std=c++17 -I/usr/include/habanalabs shim_check.cpp -o shim_check -ldl

export KIRO_TPC_LIBS=/path/libmtp_sampling_kernels_v2.so:/path/libflashinfer_gaudi_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so
./shim_check ./libkiro_tpc_shim.so ${KIRO_TPC_LIBS//:/ }
```

`shim_check` 只在主机上运行，不使用设备：它比较合并后的 GUID 与各库之和，并检查旧接口和版本号。本项目的实测结果是 3261 = 2 + 9 + 3250，无缺失。

## 使用

```bash
GC_KERNEL_PATH=/path/libkiro_tpc_shim.so
KIRO_TPC_LIBS=<与原 GC_KERNEL_PATH 相同的列表>
PT_HPU_RECIPE_CACHE_CONFIG=/path/recipe-cache,false,20480
```

服务完整预热和测速均正常。**任何一个内核库更新后都要清空缓存目录**，因为缓存的键不一定包含被转接的库的内容。
