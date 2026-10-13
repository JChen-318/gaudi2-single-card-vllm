cmake_minimum_required(VERSION 3.18)
project(mtp_sampling_v2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_program(TPC_COMPILER tpc-clang REQUIRED PATHS /opt/habanalabs/bin)
set(OBJECTS)
foreach(KERNEL rejection_race rejection_pack)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o
        COMMAND ${TPC_COMPILER} -Wall -Werror -O2 -march=gaudi2 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.c -c -o ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o
        DEPENDS kernels/${KERNEL}.c VERBATIM)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ${KERNEL}.o ${KERNEL}_embedded.o
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} VERBATIM)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o)
endforeach()
add_library(mtp_sampling_kernels_v2 SHARED host.cpp ${OBJECTS})
target_include_directories(mtp_sampling_kernels_v2 PRIVATE /usr/lib/habanatools/include /usr/include/habanalabs)
target_compile_options(mtp_sampling_kernels_v2 PRIVATE -Wall -Wextra -Werror)
target_link_options(mtp_sampling_kernels_v2 PRIVATE -Wl,-z,noexecstack)
add_executable(test_mtp_sampling_glue test_glue.cpp)
target_include_directories(test_mtp_sampling_glue PRIVATE /usr/lib/habanatools/include /usr/include/habanalabs)
target_compile_options(test_mtp_sampling_glue PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_link_libraries(test_mtp_sampling_glue PRIVATE mtp_sampling_kernels_v2)
