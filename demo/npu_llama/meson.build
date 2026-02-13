project('npu_llama', 'c', version: '1.0.0')
incdir = include_directories('include')
lib_src = ['src/npu_interface.c','src/npu_matmul.c']
# lib = library('rk3588-npu',lib_src, include_directories : incdir)
lib = static_library('rk3588-npu', lib_src, include_directories : incdir)

# Llama-2 Transformer inference with NPU-accelerated matrix multiplication
# This executable demonstrates NPU acceleration for the matmul operations in llama.c
# The NPU will be used automatically for compatible matrix dimensions, falling back to CPU otherwise
llama_npu0 = executable('llama_npu0', 'tests/llama0.c',
                       include_directories : incdir,
                       link_with : lib,
                       link_args : '-lm')

# Create a static library from llama0.c that contains all the common functions
llama_common_lib = static_library('llama_common', 'tests/llama0.c',
                                  include_directories : incdir,
                                  link_with : lib,
                                  c_args : ['-DTESTING'])

llama_npu1 = executable('llama_npu1', 'tests/llama1.c',
                       include_directories : incdir,
                       link_with : [lib, llama_common_lib],
                       link_args : '-lm')
