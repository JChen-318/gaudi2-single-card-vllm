# SPDX-License-Identifier: Apache-2.0
"""Build the Qwen4Exp NVFP4 MoE custom ops against the public Gaudi headers.

Same approach as vllm-gaudi/csrc/qwen4exp/pytorch/setup.py: only
hpu_custom_op_pt2.h from habana_frameworks.torch.utils.lib_utils.get_include_dir()
is needed, no GAUDI_PYTORCH_BRIDGE_* variables.

  cd native/qwen4exp-nvfp4-moe/pytorch
  python setup.py build_ext --build-lib <out> --build-temp <tmp>
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

setup(
    name="hpu_qwen4exp_nvfp4_moe_pt2",
    ext_modules=[
        CppExtension(
            "hpu_qwen4exp_nvfp4_moe_pt2",
            ["hpu_qwen4exp_nvfp4_moe_pt2.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            # The same soname the python side loads, so dlopen reuses the already
            # registered plugin instead of registering a second copy.
            libraries=["habana_pytorch2_plugin"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
