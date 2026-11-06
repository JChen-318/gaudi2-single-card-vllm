# SPDX-License-Identifier: Apache-2.0
"""Build against the public Gaudi custom-op headers shipped with the plugin.

Unlike csrc/deepseek_v4/pytorch (which needs the full Gaudi Bridge source
tree), this extension only uses hpu_custom_op_pt2.h from
habana_frameworks.torch.utils.lib_utils.get_include_dir() — the same approach
as csrc/flashqla_hpu. No GAUDI_PYTORCH_BRIDGE_* variables are required.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir


setup(
    name="hpu_qwen4exp_nvfp4_dequant_pt2",
    ext_modules=[
        CppExtension(
            "hpu_qwen4exp_nvfp4_dequant_pt2",
            ["hpu_qwen4exp_nvfp4_dequant_pt2.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            # The three symbols this op needs (habana::custom_op::registerUserCustomOp
            # and UserCustomOpDescriptor::{getUserCustomOpDescriptor,execute}) are
            # defined in libhabana_pytorch_backend.so, which is resident in both
            # execution modes: eager loads libhabana_pytorch2_plugin.so *and* its
            # backend dependency, lazy loads libhabana_pytorch_plugin.so plus the
            # same backend. Linking the eager plugin was mode-fragile in two ways:
            # under PT_HPU_LAZY_MODE=1 it is not resident, so dlopen either failed
            # outright (no RUNPATH) or loaded a second copy of it, whose static
            # registration aborted with "hpu::habana_d2d_memcpy_other is already
            # registered!". The backend soname always matches the resident copy, so
            # dlopen reuses it instead of registering the eager kernels again.
            libraries=["habana_pytorch_backend"],
            # Satisfy the hard DT_NEEDED without requiring LD_LIBRARY_PATH.
            runtime_library_dirs=[get_lib_dir()],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
