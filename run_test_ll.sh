#!/bin/bash

MLIR_OPT=~/llvm-project/build/bin/mlir-opt
MLIR_CPU_RUNNER=~/llvm-project/build/bin/mlir-cpu-runner
SO_DEP1=~/llvm-project/build/lib/libmlir_cuda_runtime.so
SO_DEP2=~/llvm-project/build/lib/libmlir_runner_utils.so
SO_DEP3=~/llvm-project/build/lib/libmlir_async_runtime.so
SO_DEP4=~/mlir-HashJoin/shared_stuff/shared.so

FileCheck=~/llvm-project/build/bin/FileCheck

#Input file is the first argument to the script
INPUT_FILE=$1

$MLIR_OPT  -convert-scf-to-cf  $INPUT_FILE \
    | $MLIR_OPT -gpu-kernel-outlining \
    | $MLIR_OPT -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))'\
    | $MLIR_OPT -gpu-async-region \
    | $MLIR_OPT -arith-expand \
    | $MLIR_OPT -convert-arith-to-llvm \
    | $MLIR_OPT -convert-cf-to-llvm \
    | $MLIR_OPT -convert-vector-to-llvm \
    | $MLIR_OPT -finalize-memref-to-llvm \
    | $MLIR_OPT -convert-func-to-llvm \
    | $MLIR_OPT -gpu-to-llvm \
    | $MLIR_OPT -reconcile-unrealized-casts
    # | $MLIR_CPU_RUNNER --shared-libs=$SO_DEP1 --shared-libs=$SO_DEP2 --shared-libs=$SO_DEP3 --shared-libs=$SO_DEP4 --entry-point-result=void -O0 
# | -o ./test.mlir