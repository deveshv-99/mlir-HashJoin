// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    // To initialize the tables with fixed values
    func.func @init(%rows: index, %cols: index) -> memref<?x?xi32> {

        %arr = memref.alloc(%rows, %cols) : memref<?x?xi32>
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %ci32_1 = arith.constant 1 : i32

        scf.for %i = %cidx_0 to %rows step %cidx_1 {
            scf.for %j = %cidx_0 to %cols step %cidx_1 {

                %i_i32 = arith.index_cast %i : index to i32
                %j_i32 = arith.index_cast %j : index to i32
                %val = arith.addi %i_i32, %j_i32 : i32
                memref.store %val, %arr[%i,%j] : memref<?x?xi32>
            }
        }
        return %arr: memref<?x?xi32>
    }

    gpu.module @kernels {

        // Kernel to perform nested join
        gpu.func @nested_join (%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
            %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>) 

            //---------------> Size of shared memory is fixed for now. To be changed later 
            workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>) 
            private(%temp_idx: memref<1xindex>)
            kernel 
        {
            
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            // Global_thread_index = bdim * bidx + tidx
            %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
            %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

            // Check if the thread is valid
            %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %table_x_rows : index

            scf.if %is_thread_valid {
                gpu.printf "Thread ID: %lld \n" %tidx : index
                
                // print debugging constants
                %print_thread_id = arith.constant 0: index
                %print_block_id = arith.constant 0: index

                %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
                %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
                %should_print = arith.andi %should_print_thread, %should_print_block : i1

                scf.if %should_print {
                    gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
                }

                //constants
                %cidx_0 = arith.constant 0 : index
                %cidx_1 = arith.constant 1 : index
                %cidx_2 = arith.constant 2 : index

                // Step 1: Compute the prefix sum for each thread, which is used for calculating the start index in the result array 
                // For each thread, compare its key with all keys in the smaller table

                %key1 = memref.load %table_x[%g_thread_idx, %cidx_0] : memref<?x?xi32>
                
                //%thread_sums stores the prefix sum for each thread in the block
                memref.store %cidx_0, %thread_sums[%tidx] : memref<1024xindex, 3>

                // for each key in the smaller table
                scf.for %j = %cidx_0 to %table_y_rows step %cidx_1 {
                    %key2 = memref.load %table_y[%j, %cidx_0] : memref<?x?xi32>
                    // compare the keys
                    %cmp = arith.cmpi "eq", %key1, %key2 : i32
                    // if keys match, increment the prefix sum
                    scf.if %cmp {
                        %cur_count = memref.load %thread_sums[%tidx] : memref<1024xindex, 3>
                        %new_count = arith.addi %cur_count, %cidx_1 : index
                        memref.store %new_count, %thread_sums[%tidx] : memref<1024xindex, 3>
                    }
                }
                gpu.barrier // we need this so that all the warps are done computing the thread local sums


                //Step 2: Compute the global prefix sum
                // Single threaded prefix sum

                %is_t0 = arith.cmpi "eq", %tidx, %cidx_0 : index
                scf.if %is_t0 {
                    scf.if %should_print {
                        gpu.printf "Thread start indices: [0, "
                    }
                    
                    // %temp_idx stores the current value of prefix sum
                    memref.store %cidx_0, %temp_idx[%cidx_0] : memref<1xindex> 

                    //For all threads in thread block
                    scf.for %i = %cidx_0 to %bdim step %cidx_1 {
                        %g_thread_index = arith.addi %g_thread_offset_in_blocks, %i : index

                        %is_valid = arith.cmpi "ult", %g_thread_index, %table_x_rows : index
                        scf.if %is_valid{

                            %cur_count = memref.load %thread_sums[%i] : memref<1024xindex, 3>
                            %cur_idx = memref.load %temp_idx[%cidx_0] : memref<1xindex>
                            %next_index = arith.addi %cur_idx, %cur_count : index

                            //thread_sums[i] stores the starting index to write from for thread i, which is 0 for thread 0
                            memref.store %cur_idx, %thread_sums[%i] : memref<1024xindex, 3>
                            memref.store %next_index, %temp_idx[%cidx_0] : memref<1xindex>

                            scf.if %should_print {
                                gpu.printf "%ld, " %next_index : index
                            }
                        }
                    }

                    scf.if %should_print {
                        gpu.printf "]\n"
                    }

                    //Compute global block offset
                    %total_elements = memref.load %temp_idx[%cidx_0] : memref<1xindex>
                    %total_elements_i32 = arith.index_cast %total_elements : index to i32
                    
                    %cur_block_offset = memref.atomic_rmw addi %total_elements_i32, %gblock_offset[%cidx_0] : (i32, memref<1xi32>) -> i32
                    %cur_block_offset_idx = arith.index_cast %cur_block_offset : i32 to index

                    memref.store %cur_block_offset_idx, %b_block_offset[%cidx_0] : memref<1xindex, 3>

                    scf.if %should_print {
                        gpu.printf "Current block# %ld offset: %ld, total elements: %ld\n" %bidx, %cur_block_offset_idx, %total_elements : index, index, index
                    }
                }

                gpu.barrier // other threads need to wait until the prefix sum is complete

                // Step 3: Each thread needs to store its value in the result array
                // The start index are loaded from the thread_sums array
                // The current index is stored in the temp_idx array for each thread
                // It is incremented after each store that matches the keys

                %cur_block_offset = memref.load %b_block_offset[%cidx_0] : memref<1xindex, 3>
                %cur_thread_offset = memref.load %thread_sums[%tidx] : memref<1024xindex, 3>
                %start_index = arith.addi %cur_block_offset, %cur_thread_offset : index
                memref.store %start_index, %temp_idx[%cidx_0] : memref<1xindex>

                scf.if %should_print {
                    gpu.printf "Block %ld, thread ID: %ld: start_index: %ld \n" %bidx, %tidx, %start_index: index, index, index
                }

                // for each key in the smaller table
                scf.for %i = %cidx_0 to %table_y_rows step %cidx_1 {
                    %key2 = memref.load %table_y[%i, %cidx_0] : memref<?x?xi32>
                    // compare the keys
                    %cmp = arith.cmpi "eq", %key1, %key2 : i32
                    // if keys match, store the values in the result array
                    scf.if %cmp {
                        // load the current index to write to in the result array
                        %cur_idx = memref.load %temp_idx[%cidx_0] : memref<1xindex>

                        // store the values from table x in d_result[0 to table_x_cols] array
                        scf.for %j = %cidx_0 to %table_x_cols step %cidx_1 {
                            %val1 = memref.load %table_x[%g_thread_idx, %j] : memref<?x?xi32>
                            memref.store %val1, %d_result[%cur_idx, %j] : memref<?x?xi32>
                        }
                        // just to find the index in column for result array
                        %table_x_cols_ = arith.subi %table_x_cols, %cidx_1 : index

                        // store the values from table y in d_result[table_x_cols to table_x_cols+table_y_cols] array
                        // start j from 1, to avoid storing the key again
                        scf.for %j = %cidx_1 to %table_y_cols step %cidx_1 {
                            %res_y_col = arith.addi %table_x_cols_, %j : index
                            %val2 = memref.load %table_y[%i, %j] : memref<?x?xi32>
                            memref.store %val2, %d_result[%cur_idx, %res_y_col] : memref<?x?xi32>
                        }
                        // update the next index for the result array
                        %next_idx = arith.addi %cur_idx, %cidx_1 : index
                        memref.store %next_idx, %temp_idx[%cidx_0] : memref<1xindex>
                    }
                }
            }
            
            gpu.return
        }
    }
    
    func.func @main() {

        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32

        // Table sizes have to be passed as arguments later on, with the number of columns for each table.
        // For our specific example, part_table is table_1 and line_order is table_2 
        %table_1_rows = arith.constant 20 : index
        %table_2_rows = arith.constant 20 : index
        
        %table_1_cols = arith.constant 3 : index
        %table_2_cols = arith.constant 2 : index

        //Initialize the tables to fixed values for now.. 
        %h_table_1 = call @init(%table_1_rows, %table_1_cols) : (index,index) -> memref<?x?xi32>
        %h_table_2 = call @init(%table_2_rows, %table_2_cols) : (index,index) -> memref<?x?xi32>

        // Allocate device memory for the tables
        %d_table_1 = gpu.alloc(%table_1_rows, %table_1_cols) : memref<?x?xi32>
        gpu.memcpy %d_table_1, %h_table_1 : memref<?x?xi32>, memref<?x?xi32>
        %d_table_2 = gpu.alloc(%table_2_rows, %table_2_cols) : memref<?x?xi32>
        gpu.memcpy %d_table_2, %h_table_2 : memref<?x?xi32>, memref<?x?xi32>

        // //-------------> Allocate result array to contain number of rows as size of table1*table2... Need something better than this?
        %result_rows = arith.muli %table_1_rows, %table_2_rows : index
        // Result columns will be (sum of columns of both tables - 1 (for the duplicate key))
        %result_cols_ = arith.addi %table_1_cols, %table_2_cols : index 
        %result_cols = arith.subi %result_cols_, %cidx_1 : index 

        %d_result = gpu.alloc(%result_rows, %result_cols) : memref<?x?xi32>

        // //-------------> Keep threads per block constant at 1024.. Need to change this?
        %num_threads_per_block = arith.constant 1024 : index

        // //-------------> Keep items per thread constant at 1.. Not sure about this either
        %items_per_thread = arith.constant 1 : index

        //global variable for all blocks
        %gblock_offset = gpu.alloc() : memref<1xi32>

        //Whichever table is smaller, we use that for comparison (as the inner loop)
        // i.e. the larger table is allocated to threads (outer loop)
        %table_1_or_2_as_inner = arith.cmpi "ult", %table_1_rows, %table_2_rows : index

        //Number of threads would be the number of rows in the larger table
        %total_threads = arith.select %table_1_or_2_as_inner, %table_2_rows, %table_1_rows : index

        // To calculate the number of blocks needed, perform ceil division: num_blocks = (total_threads + num_threads_per_block - 1) / num_threads_per_block
        // TODO: arith.ceildivui gives errors which i cant figure out. so using the above thing instead..
        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index


        //defining parameters to be passed to the kernel
        // Table x is the outer loop, which will be assigned to threads. and Table y is the inner loop used for comparison
        %table_x = arith.select %table_1_or_2_as_inner, %d_table_2, %d_table_1 : memref<?x?xi32>
        %table_y = arith.select %table_1_or_2_as_inner, %d_table_1, %d_table_2 : memref<?x?xi32>
        
        %table_x_rows = arith.select %table_1_or_2_as_inner, %table_2_rows, %table_1_rows : index
        %table_x_cols = arith.select %table_1_or_2_as_inner, %table_2_cols, %table_1_cols : index
        %table_y_rows = arith.select %table_1_or_2_as_inner, %table_1_rows, %table_2_rows : index
        %table_y_cols = arith.select %table_1_or_2_as_inner, %table_1_cols, %table_2_cols : index

        gpu.launch_func @kernels::@nested_join
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
            %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>)

        // copy the result column from the device to host
        %h_result = memref.alloc(%result_rows, %result_cols) : memref<?x?xi32>
        gpu.memcpy %h_result, %d_result : memref<?x?xi32>, memref<?x?xi32>

        // print the result
        %dst = memref.cast %h_result : memref<?x?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        // deallocate the arrays
        memref.dealloc %h_table_1 : memref<?x?xi32>
        memref.dealloc %h_table_2 : memref<?x?xi32>
        memref.dealloc %h_result : memref<?x?xi32>
        gpu.dealloc %d_table_1 : memref<?x?xi32>
        gpu.dealloc %d_table_2 : memref<?x?xi32>
        gpu.dealloc %d_result : memref<?x?xi32>
        gpu.dealloc %gblock_offset : memref<1xi32>
        
        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
}