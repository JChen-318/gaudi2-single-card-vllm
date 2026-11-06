# SPDX-License-Identifier: Apache-2.0
# NVFP4 MoE TPC library for Gaudi2 (decode GEMV + prefill bucketing).
#
#   cmake -S native/qwen4exp-nvfp4-moe -B build/native/qwen4exp-nvfp4-moe -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native/qwen4exp-nvfp4-moe --parallel
#
# Builds libqwen4exp_nvfp4_moe_kernels.so (TPC ELFs embedded, no device needed),
# the host geometry test, and - when the Habana TPC simulator libraries are
# installed - moe_sim, which runs the kernels in the host simulator.
cmake_minimum_required(VERSION 3.18)
project(qwen4exp_nvfp4_moe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC headers")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API headers")

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "kernels/*.c")
file(GLOB KERNEL_HEADERS CONFIGURE_DEPENDS "include/*.h")
set(EMBEDDED_OBJECTS)
foreach(SOURCE IN LISTS KERNEL_SOURCES)
    get_filename_component(NAME "${SOURCE}" NAME_WE)
    set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")
    set(EMBEDDED "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_embedded.o")
    add_custom_command(OUTPUT "${OBJECT}"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2
            -I${TPC_INCLUDE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/include
            "${SOURCE}" -c -o "${OBJECT}"
        DEPENDS "${SOURCE}" ${KERNEL_HEADERS} VERBATIM)
    add_custom_command(OUTPUT "${EMBEDDED}"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            "./${NAME}.o" "${NAME}_embedded.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS "${OBJECT}" VERBATIM)
    set_source_files_properties("${EMBEDDED}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND EMBEDDED_OBJECTS "${EMBEDDED}")
endforeach()

file(GLOB HOST_SOURCES CONFIGURE_DEPENDS "host/*.cpp")
# pytorch/*.cpp builds separately via pytorch/setup.py against the installed
# Gaudi Bridge headers (habana_frameworks).
add_library(qwen4exp_nvfp4_moe_kernels SHARED ${HOST_SOURCES} ${EMBEDDED_OBJECTS})
target_include_directories(qwen4exp_nvfp4_moe_kernels PRIVATE
    host include "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
target_compile_options(qwen4exp_nvfp4_moe_kernels PRIVATE -Wall -Wextra -fno-strict-aliasing)
target_link_options(qwen4exp_nvfp4_moe_kernels PRIVATE -Wl,-z,noexecstack -Wl,--no-undefined)
target_link_libraries(qwen4exp_nvfp4_moe_kernels PRIVATE ${CMAKE_DL_LIBS})

enable_testing()
add_executable(qwen4exp_nvfp4_moe_geometry_test tests/glue_geometry.cpp)
target_include_directories(qwen4exp_nvfp4_moe_geometry_test PRIVATE
    "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}" include)
target_link_libraries(qwen4exp_nvfp4_moe_geometry_test PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME qwen4exp_nvfp4_moe_geometry COMMAND qwen4exp_nvfp4_moe_geometry_test
    $<TARGET_FILE:qwen4exp_nvfp4_moe_kernels>)

# Scalar C reference for the decode GEMV math: E2M1 table x E4M3 scale in double
# precision, double accumulation, scale2 after the sum. No TPC compiler, no
# device. validation/qwen4exp/test_nvfp4_moe_reference.py compares its numpy
# emulation of the kernel against this program (and compiles the same source
# itself when this binary has not been built).
add_executable(qwen4exp_nvfp4_moe_reference tests/nvfp4_moe_gemv_reference.c)
target_compile_options(qwen4exp_nvfp4_moe_reference PRIVATE -Wall -Wextra)
target_link_libraries(qwen4exp_nvfp4_moe_reference PRIVATE m)

# TPC simulator (host only, no device): the same libraries the gdn-fused-kernel
# harness uses. Optional so the library still builds where they are missing.
find_library(TPCSIM_LIB tpcsim_shared PATHS "${TPC_INCLUDE_DIR}/.." NO_DEFAULT_PATH)
find_library(TPCTEST_LIB tpc_tests_core_ext PATHS "${TPC_INCLUDE_DIR}/.." NO_DEFAULT_PATH)
if(TPCSIM_LIB AND TPCTEST_LIB AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/moe_sim.cpp")
    add_executable(qwen4exp_nvfp4_moe_sim
        tests/moe_sim.cpp
        host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp
        host/qwen4exp_nvfp4_moe_count_i32_gaudi2.cpp
        host/qwen4exp_nvfp4_moe_prefix_i32_gaudi2.cpp
        host/qwen4exp_nvfp4_moe_scatter_i32_gaudi2.cpp
        ${EMBEDDED_OBJECTS})
    target_include_directories(qwen4exp_nvfp4_moe_sim PRIVATE
        host include "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
    target_link_libraries(qwen4exp_nvfp4_moe_sim PRIVATE
        ${TPCTEST_LIB} ${TPCSIM_LIB} ${CMAKE_DL_LIBS})
    target_link_options(qwen4exp_nvfp4_moe_sim PRIVATE
        -Wl,-rpath,${TPC_INCLUDE_DIR}/..)
    add_test(NAME qwen4exp_nvfp4_moe_simulator COMMAND qwen4exp_nvfp4_moe_sim)
    set_tests_properties(qwen4exp_nvfp4_moe_simulator PROPERTIES TIMEOUT 600)

    # Per-call latency estimate at the checkpoint's real layer shapes. Same ELF
    # and harness: it measures K=2560/N=1280 and K=640/N=2560 at S=10 and S=1,
    # so it runs for about a minute in the simulator. Cycles are instruction
    # counts at 1 IPC; see the file header for the conversion assumptions.
    add_executable(qwen4exp_nvfp4_moe_latency
        tests/qwen4exp_nvfp4_moe_latency.cpp
        host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp
        ${EMBEDDED_OBJECTS})
    target_include_directories(qwen4exp_nvfp4_moe_latency PRIVATE
        host include "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
    target_link_libraries(qwen4exp_nvfp4_moe_latency PRIVATE
        ${TPCTEST_LIB} ${TPCSIM_LIB} ${CMAKE_DL_LIBS})
    target_link_options(qwen4exp_nvfp4_moe_latency PRIVATE
        -Wl,-rpath,${TPC_INCLUDE_DIR}/..)
    add_test(NAME qwen4exp_nvfp4_moe_latency COMMAND qwen4exp_nvfp4_moe_latency)
    set_tests_properties(qwen4exp_nvfp4_moe_latency PROPERTIES TIMEOUT 900)
endif()
