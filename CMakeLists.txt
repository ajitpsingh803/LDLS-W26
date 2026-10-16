cmake_minimum_required(VERSION 3.14)
project(ldls_structured_kv CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
if (NOT EXISTS ${LLAMA_DIR}/include/llama.h)
    message(FATAL_ERROR "llama.cpp not found in ${LLAMA_DIR}; run scripts/setup_llama_cpp.sh first")
endif()

# CPU-only build of libllama; we only need the library, not the llama.cpp tools.
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

# On the course machine (Sapphire Rapids, Linux 5.15) ggml's AMX matmul kernels die with SIGILL
# (`tilezero`) once prompts reach a few hundred tokens. Compiling ggml-cpu without AMX keeps
# AVX-512/VNNI and the repacked Q4_K kernels. These flags come after -march=native, so they win.
option(LDLS_DISABLE_AMX "build ggml-cpu without AMX kernels (x86 only)" ON)
if (LDLS_DISABLE_AMX AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(ggml-cpu PRIVATE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16)
endif()

set(LDLS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/driver/src ${LLAMA_DIR}/vendor)

add_executable(ldls_tokenize driver/src/ldls_tokenize.cpp)
target_include_directories(ldls_tokenize PRIVATE ${LDLS_INCLUDES})
target_link_libraries(ldls_tokenize PRIVATE llama)

add_executable(ldls_driver
    driver/src/main.cpp
    driver/src/structured_doc.cpp
    driver/src/engine.cpp
)
target_include_directories(ldls_driver PRIVATE ${LDLS_INCLUDES})
target_link_libraries(ldls_driver PRIVATE llama)

foreach(t ldls_tokenize ldls_driver)
    set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
