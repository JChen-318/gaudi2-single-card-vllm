# HCL
HCL is Habana Communication library.<br />
Internally it implements multi-device and multi-node communication and reduction primitives, optimized for Habana devices.<br />
It exposes HCCL APIs described in:  https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/C_API.html

## Contents
C++ project which includes HCL relevant source code.<br />
Building this project should generate 'hcl.so' library

## Licensing
Copyright (c) 2022 Habana Labs, Ltd.<br />
SPDX-License-Identifier: Apache-2.0

## Build
HCL project can be built using the following command:<br />

   - Go to HCL source directory
   ```
   cd <target-directory>/hcl/src
   ```
   - Run cmake. set HCL_SRC_PKG_DIR to the source package directory and HCL_LIB_DIR to the HCL libs directory
   ```
   HCL_SRC_PKG_DIR=<target-directory> HCL_LIB_DIR=<hcl-lib-dir> cmake .
   ```
   - Build the project
   ```
   make
   ```
   Note that HCL lib will be created under /usr/lib/habanalabs
   