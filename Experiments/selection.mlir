// To run this file:
// LINGODB_DIR=/home/ssd/workdesk/lingo-db
// LLVM_BUILD_DIR=$LINGODB_DIR/llvm-project/build
// BIN_DIR=$LLVM_BUILD_DIR/bin
// LIB_DIR=$LLVM_BUILD_DIR/lib
// mlir_opt=$BIN_DIR/mlir-opt
// mlir_cpurunner=$BIN_DIR/mlir-cpu-runner

// $mlir_opt $1 \
//   -gpu-kernel-outlining \
//   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
//   --convert-scf-to-cf -gpu-async-region -gpu-to-llvm \
// | $mlir_cpurunner \
//   --shared-libs=$LIB_DIR/libmlir_cuda_runtime.so \
//   --shared-libs=$LIB_DIR/libmlir_runner_utils.so \
//   --entry-point-result=void

module attributes {gpu.container_module} {

func.func @init(%size: index) -> memref<?xf32> {
  %arr = memref.alloc(%size) : memref<?xf32>
  %c_idx = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c_idx to %size step %c1 {
    %t = arith.index_cast %i : index to i32
    %r = arith.sitofp %t : i32 to f32
    memref.store %r, %arr[%i] : memref<?xf32>
  }
  return %arr: memref<?xf32>
}

gpu.module @query_execution{

gpu.func @run_selection(%d_arrayA: memref<?xf32>, %d_result: memref<?xf32>, %array_len: index, %gblock_offset: memref<1xi32>, %items_per_thread: index)
  workgroup(%thread_sums : memref<128xindex, 3>, %b_block_offset: memref<1xindex, 3>) // shared memory. TODO: Make it take the size as an argument
  private(%temp_idx : memref<1xindex>)
  kernel
{
    // standard code: result thread id r_itd = bdim * bidx + tid
    %bdim = gpu.block_dim x
    %bidx = gpu.block_id x
    %tid = gpu.thread_id x
    %num_threads = arith.muli %bidx, %bdim : index
    %boffset = arith.muli %num_threads, %items_per_thread : index
    %elem_offset = arith.muli %tid, %items_per_thread : index
    %thread_start_idx = arith.addi %boffset, %elem_offset : index

    // print debugging constants
    %print_thread_id = arith.constant 0: index
    %print_block_id = arith.constant 1: index
    %should_print_thread = arith.cmpi "eq", %tid, %print_thread_id : index
    %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
    %should_print = arith.andi %should_print_thread, %should_print_block : i1

    scf.if %should_print {
      gpu.printf "Block ID: %ld, Thread ID: %ld, thread_start_idx: %ld\n" %bidx, %tid, %thread_start_idx : index, index, index
    }

    // constants
    %c1_idx = arith.constant 1 : index
    %compare_val = arith.constant 80.0 : f32
    %c0_idx = arith.constant 0 : index
    %c32 = arith.constant 32 : i32
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %c128_idx = arith.constant 128 : index

    // for each thread, check for the predicate d_arrayA[r_tid] < 10
    // step 1: compute the prefix sum. count the number of elements that pass the predicate
    scf.for %i = %c0_idx to %items_per_thread step %c1_idx {
      %idx = arith.addi %thread_start_idx, %i : index // cur element index for the current thread
      %val = memref.load %d_arrayA[%idx] : memref<?xf32>
      %condition = arith.cmpf "olt", %val, %compare_val : f32 // ordered less than 10
      scf.if %condition {
        %cur_count = memref.load %thread_sums[%tid] : memref<128xindex, 3>
        %new_count = arith.addi %cur_count, %c1_idx : index
        memref.store %new_count, %thread_sums[%tid] : memref<128xindex, 3>
      }
    }

    gpu.barrier // we need this so that all the warps are done computing the thread local sums

    // step 2: each thread needs to know where in the result array to store its value
    // TODO: We are doing a single threaded prefix sum here. Sufficient for now. Nothing new to learn doing the
    // parallel prefix sum for now.

    %is_t0 = arith.cmpi "eq", %tid, %c0_idx : index

    scf.if %is_t0 {
      %end_index = arith.subi %bdim, %c1_idx : index
      scf.if %should_print {
        gpu.printf "Thread start indices: [0, "
      }

      memref.store %c0_idx, %temp_idx[%c0_idx] : memref<1xindex> // start with the first thread
      scf.for %i = %c0_idx to %bdim step %c1_idx {
        %cur_idx = memref.load %temp_idx[%c0_idx] : memref<1xindex>
        %cur_count = memref.load %thread_sums[%i] : memref<128xindex, 3>
        %next_index = arith.addi %cur_idx, %cur_count : index
        memref.store %next_index, %temp_idx[%c0_idx] : memref<1xindex> // store the current index in a temp variable
        memref.store %cur_idx, %thread_sums[%i] : memref<128xindex, 3>
        scf.if %should_print {
          gpu.printf "%ld, " %next_index : index
        }
      }

      scf.if %should_print {
        gpu.printf "]\n"
      }

      // Compute global block offset
      %total_elements = memref.load %temp_idx[%c0_idx] : memref<1xindex>
      %total_elements_i32 = arith.index_cast %total_elements : index to i32
      %cur_block_offset = memref.atomic_rmw addi %total_elements_i32, %gblock_offset[%c0_idx] : (i32, memref<1xi32>) -> i32
      %cur_block_offset_idx = arith.index_cast %cur_block_offset : i32 to index
      memref.store %cur_block_offset_idx, %b_block_offset[%c0_idx] : memref<1xindex, 3>
      // scf.if %should_print {
        gpu.printf "Current block# %ld offset: %ld, total elements: %ld\n" %bidx, %cur_block_offset_idx, %total_elements : index, index, index
      // }

    }

    gpu.barrier // other threads need to wait until the prefix sum is complete

    // Step 3: Each thread needs to store its value in the result array
    // The starting index are loaded from the thread_sums array
    // The current index is stored in the temp_idx array for each thread
    // It is incremented after each store that passes the predicate
    %cur_block_offset = memref.load %b_block_offset[%c0_idx] : memref<1xindex, 3>
    %cur_thread_offset = memref.load %thread_sums[%tid] : memref<128xindex, 3>
    %start_index = arith.addi %cur_block_offset, %cur_thread_offset : index
    memref.store %start_index, %temp_idx[%c0_idx] : memref<1xindex>

    scf.if %should_print {
      gpu.printf "Block %ld, thread ID: %ld: start_index: %ld \n" %bidx, %tid, %start_index: index, index, index
    }

    scf.for %i = %c0_idx to %items_per_thread step %c1_idx {
      %idx = arith.addi %thread_start_idx, %i : index // cur element index for the current thread
      %val = memref.load %d_arrayA[%idx] : memref<?xf32>
      %condition = arith.cmpf "olt", %val, %compare_val : f32 // ordered less than 10
      scf.if %condition {
        %cur_idx = memref.load %temp_idx[%c0_idx] : memref<1xindex>
        memref.store %val, %d_result[%cur_idx] : memref<?xf32>
        scf.if %should_print {
          gpu.printf "bid: %ld, tid: %ld, val: %f storing at result index %ld\n" %bidx, %tid, %val, %cur_idx : index, index, f32, index
        }
        // update the next index in the result array
        %next_idx = arith.addi %cur_idx, %c1_idx : index
        memref.store %next_idx, %temp_idx[%c0_idx] : memref<1xindex>
      }
    }
    gpu.return
}

}

func.func @main() {

  // some constants
  %c_arr_len = arith.constant 128 : index

    // initialize the arrays to random numbers
  %h_arrayA = call @init(%c_arr_len) : (index) -> memref<?xf32>

  // copy the arrays to the GPU
  %d_arrayA = gpu.alloc(%c_arr_len) : memref<?xf32>
  gpu.memcpy %d_arrayA, %h_arrayA : memref<?xf32>, memref<?xf32>

  // create an array of one element to hold the result
  %d_result = gpu.alloc(%c_arr_len) : memref<?xf32>

  // compute items per thread
  %num_blocks = arith.constant 2 : index
  %num_threads = arith.constant 32: index // num threads per block
  %total_threads = arith.muli %num_blocks, %num_threads : index
  %items_per_thread = arith.divui %c_arr_len, %total_threads : index

  // launch the kernel
  %one = arith.constant 1 : index
  %gblock_offset = gpu.alloc() : memref<1xi32>
  gpu.launch_func @query_execution::@run_selection
    blocks in (%num_blocks, %one, %one)
    threads in (%num_threads, %one, %one)
    args (%d_arrayA: memref<?xf32>, %d_result: memref<?xf32>, %c_arr_len: index, %gblock_offset: memref<1xi32>, %items_per_thread: index)

  // copy the result column from the GPU
  %h_result = memref.alloc(%c_arr_len) : memref<?xf32>
  gpu.memcpy %h_result, %d_result : memref<?xf32>, memref<?xf32>

  // print the result
  %dst = memref.cast %h_result : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%dst) : (memref<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
}