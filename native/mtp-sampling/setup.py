# SPDX-License-Identifier: Apache-2.0
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="mtp_sampling_v2",
    ext_modules=[
        CppExtension(
            "mtp_sampling_v2",
            ["op.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            libraries=["habana_pytorch2_plugin"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
