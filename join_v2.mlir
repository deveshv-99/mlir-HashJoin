// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {
    memref.global constant @buildRelationRows : memref<1xindex> = dense<[100000000]>
    memref.global constant @probeRelationRows : memref<1xindex> = dense<[100000000]>

    memref.global constant @hashTableSize : memref<1xindex> = dense<[10000000]>

    memref.global constant @numberOfThreadsPerBlock : memref<1xindex> = dense<[256]>

    // Function to print the contents of a single memref value
    func.func @debugI32(%inputValue: i32) {
        %allocatedMemRef = memref.alloc() : memref<i32>
        memref.store %inputValue, %allocatedMemRef[]: memref<i32>
        %castMemRef = memref.cast %allocatedMemRef :  memref<i32> to memref<*xi32>

        func.call @printMemrefI32(%castMemRef): (memref<*xi32>) -> ()

        memref.dealloc %allocatedMemRef : memref<i32>
        return
    }

    // Allocate GPU memory for the hash table and the buckets
    func.func @allocateHashTable(%numTuples : index, %hashTableSize : index) 
        -> ( memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>) {
            
        // Allocate linked_list 
        %linkedListKey = gpu.alloc(%numTuples) : memref<?xi32>
        %linkedListRowId = gpu.alloc(%numTuples) : memref<?xindex>
        %linkedListnextIndex = gpu.alloc(%numTuples) : memref<?xindex>

        // Allocate hash table
        %hashTablePointers = gpu.alloc(%hashTableSize) : memref<?xi32>
        
        // Return all of the allocated memory
        return %linkedListKey, %linkedListRowId, %linkedListnextIndex, %hashTablePointers
            : memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>
    }

    // // To calculate the number of blocks needed, perform ceil division:
    // // numberOfBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock
    func.func @calculateNumberOfBlocks(%totalThreads : index, %threadsPerBlock: index) -> index {
        
        // Constants
        %oneIndex = arith.constant 1 : index

        %adjustedTotal = arith.addi %totalThreads, %threadsPerBlock : index
        %adjustedTotalMinusOne = arith.subi %adjustedTotal, %oneIndex : index
        %numberOfBlocks = arith.divui %adjustedTotalMinusOne, %threadsPerBlock : index
        return %numberOfBlocks : index
    }

    func.func @initializeHashTable(%hashTableSize : index, %hashTablePointers : memref<?xi32>) {
            
        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index

        %threadsPerBlockMemeref = memref.get_global @numberOfThreadsPerBlock : memref<1xindex>
        %threadsPerBlock = memref.load %threadsPerBlockMemeref[%zeroIndex] : memref<1xindex>

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%hashTableSize, %threadsPerBlock) : (index, index) -> index

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelHashTable::@initializeHT
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex)
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%hashTableSize : index, %hashTablePointers : memref<?xi32>)

        func.call @endTimer() : () -> ()

        return
    }

    func.func @buildTable(%buildRelation : memref<?xi32>, %buildRelationRows : index, %hashTablePointers : memref<?xi32>,
        %linkedListKey : memref<?xi32>, %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>, %hashTableSize : i32) {
        
        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index
        %zeroI32 = arith.constant 0 : i32

        //global variable for next free index in linked_list
        %freeIndex = gpu.alloc() : memref<1xi32>

        //store 0 initially
        gpu.memset %freeIndex, %zeroI32 : memref<1xi32>, i32

        %threadsPerBlockMemeref = memref.get_global @numberOfThreadsPerBlock : memref<1xindex>
        %threadsPerBlock = memref.load %threadsPerBlockMemeref[%zeroIndex] : memref<1xindex>

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%buildRelationRows, %threadsPerBlock) : (index, index) -> index


        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelBuild::@build
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%buildRelation : memref<?xi32>, %buildRelationRows : index, %hashTablePointers : memref<?xi32>, %linkedListKey : memref<?xi32>,
            %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>, %hashTableSize : i32)

        func.call @endTimer() : () -> ()
        return

    }

    func.func @countRows(%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,%linkedListKey : memref<?xi32>,
        %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  %prefixSumArray : memref<?xindex>, %hashTableSize : i32) -> (index) {

        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index

        %zeroI32 = arith.constant 0 : i32
        
        %threadsPerBlockMemeref = memref.get_global @numberOfThreadsPerBlock : memref<1xindex>
        %threadsPerBlock = memref.load %threadsPerBlockMemeref[%zeroIndex] : memref<1xindex>

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%probeRelationRows, %threadsPerBlock) : (index, index) -> index

        // Global variable for block offset
        %globalBlockOffset = gpu.alloc() : memref<1xi32>
        gpu.memset %globalBlockOffset, %zeroI32 : memref<1xi32>, i32

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelCount::@count 
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,
            %prefixSumArray : memref<?xindex>, %globalBlockOffset : memref<1xi32>, %hashTableSize : i32)

        func.call @endTimer() : () -> ()
        
        // After the kernel has executed, the globalBlockOffset contains the size of the result array
        %hostGlobalBlockOffset = memref.alloc() : memref<1xi32>
        gpu.memcpy %hostGlobalBlockOffset, %globalBlockOffset : memref<1xi32>, memref<1xi32>
        
        %resultSize = memref.load %hostGlobalBlockOffset[%zeroIndex] : memref<1xi32>
        %resultSizeIndex = arith.index_cast %resultSize : i32 to index

        return %resultSizeIndex : index
    }

    func.func @probeRelation(%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTableSize : i32, %hashTablePointers : memref<?xi32>, 
        %linkedListKey : memref<?xi32>, %linkedListRowId : memref<?xindex> , %linkedListnextIndex : memref<?xindex>, %prefixSumArray : memref<?xindex>,
        %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>){
        
        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index

        %zeroI32 = arith.constant 0 : i32
        
        %threadsPerBlockMemeref = memref.get_global @numberOfThreadsPerBlock : memref<1xindex>
        %threadsPerBlock = memref.load %threadsPerBlockMemeref[%zeroIndex] : memref<1xindex>

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%probeRelationRows, %threadsPerBlock) : (index, index) -> index

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelProbe::@probe

        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTableSize : i32, %hashTablePointers : memref<?xi32>,
            %linkedListKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,
            %prefixSumArray : memref<?xindex>, %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>)

        func.call @endTimer() : () -> ()
        return
    }
    
    gpu.module @kernelHashTable {

        gpu.func @initializeHT(%hashTableSize : index, %hashTablePointers : memref<?xi32>) 
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // globalThreadIndex = blockDim * blockId + threadId
            %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
            %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

            // check  if the thread is valid
            %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %hashTableSize : index

            scf.if %isThreadValid {

                // Set the hashTablePointers to -1
                %negOneI32 = arith.constant -1 : i32
                memref.store %negOneI32, %hashTablePointers[%globalThreadIndex] : memref<?xi32>
    
            }
            gpu.return 
        }
    }
    gpu.module @kernelBuild {

        func.func @hash(%probeKey : i32, %hashTableSize : i32) -> i32{
            //%modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %hashTableSize : i32
            return %hashValue : i32
        }


        func.func @insertNodeInHashTable(%key : i32, %hashTablePointers : memref<?xi32>, %linkedListKey : memref<?xi32>,
            %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>, %globalThreadIndex : index, %hashTableSize : i32) {

            // constants
            %negOneIndex = arith.constant -1 : index
            %zeroIndex = arith.constant 0 : index 

            %negOneI32 = arith.constant -1 : i32
            %oneI32 = arith.constant 1 : i32

            // The free index at which the node is being modified
            %indexI32 = memref.atomic_rmw addi %oneI32, %freeIndex[%zeroIndex] : (i32, memref<1xi32>) -> i32

            // gpu.printf " indexI32: %d \n" %indexI32 : i32
            // cast the i32 to index
            %index = arith.index_cast %indexI32 : i32 to index

            // Insert key and rowID into the new node
            memref.store %key, %linkedListKey[%index] : memref<?xi32>
            memref.store %globalThreadIndex, %linkedListRowId[%index] : memref<?xindex>

            // compute the hash value
            %hashValueI32 = func.call @hash(%key, %hashTableSize) : (i32, i32) -> i32
            // gpu.printf "hashValue: %d \n" %hashValueI32 : i32
            // cast the hash value to index
            %hashValue = arith.index_cast %hashValueI32 : i32 to index

            %cmp_val = memref.load %hashTablePointers[%hashValue] : memref<?xi32>

            // implement memref.rmw to update the hash table
            %oldIndexI32 = memref.atomic_rmw assign %indexI32, %hashTablePointers[%hashValue] : (i32, memref<?xi32>) -> i32
            // cast the index to i32
            %oldIndex = arith.index_cast %oldIndexI32 : i32 to index
            memref.store %oldIndex, %linkedListnextIndex[%index] : memref<?xindex>

            return
        }

        gpu.func @build(%buildRelation : memref<?xi32>, %buildRelationRows : index,
                %hashTablePointers : memref<?xi32>, %linkedListKey : memref<?xi32>, %linkedListRowId : memref<?xindex>, 
                %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>, %hashTableSize : i32)
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // globalThreadIndex = blockDim * blockId + threadId
            %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
            %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

            // checkBucketNotEmpty if the thread is valid
            %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %buildRelationRows : index

            scf.if %isThreadValid {

                %key = memref.load %buildRelation[%globalThreadIndex] : memref<?xi32>
                
                func.call @insertNodeInHashTable(%key, %hashTablePointers, %linkedListKey, %linkedListRowId, %linkedListnextIndex, %freeIndex, %globalThreadIndex, %hashTableSize) 
                : (i32, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index, i32) -> ()
               
            }
            
            gpu.return
        }
    }

    gpu.module @kernelCount {

        func.func @hash(%probeKey : i32, %hashTableSize : i32) -> i32{
            //%modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %hashTableSize : i32
            return %hashValue : i32
        }

        gpu.func @count (%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  
            %prefixSumArray : memref<?xindex>, %globalBlockOffset : memref<1xi32>, %hashTableSize : i32)
            // sharedMEmPrefixSum size should be the block size
            workgroup(%sharedMemPrefixSum : memref<1024xindex, 3>, %sharedMemTotalSum : memref<1xindex, 3>)

            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // globalThreadIndex = blockDim * blockId + threadId
            %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
            %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

            // check if the thread is valid
            %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %probeRelationRows : index

            scf.if %isThreadValid {
                //constants
                %negOneIndex = arith.constant -1 : index 
                %zeroIndex = arith.constant 0 : index
                %oneIndex = arith.constant 1 : index

                %negOneI32 = arith.constant -1 : i32

                // Initialize the prefixSumArray sum to 0  
                memref.store %zeroIndex, %prefixSumArray[%globalThreadIndex] : memref<?xindex>

                // %sharedMemPrefixSum stores the prefixSumArray sum for each thread in the block
                memref.store %zeroIndex, %sharedMemPrefixSum[%threadId] : memref<1024xindex, 3>

                // Step 1: Compute the prefixSumArray sum for each thread, which is used for calculating the start index in the result array 
                // For each thread, compare its probeKey with all probeKeys in the relevant hash bucket chain

                %probeKey = memref.load %probeRelation[%globalThreadIndex] : memref<?xi32>

                %hashValueI32 = func.call @hash(%probeKey, %hashTableSize) : (i32, i32) -> i32
                %hashValue = arith.index_cast %hashValueI32 : i32 to index

                // To get the first node in the linked list
                %currentBucketIndexI32 = memref.load %hashTablePointers[%hashValue] : memref<?xi32>
                
                // check if the list is not empty
                %checkBucketNotEmpty = arith.cmpi "ne", %currentBucketIndexI32, %negOneI32 : i32 
                scf.if %checkBucketNotEmpty{
    
                    %currentBucketIndex = arith.index_cast %currentBucketIndexI32 : i32 to index

                    %currentCount = memref.load %sharedMemPrefixSum[%threadId] : memref<1024xindex, 3>
                    // do while loop to iterate over the linked list
                    %res, %finalCount = scf.while (%arg1 = %currentBucketIndex, %arg2 = %currentCount) :(index, index) -> (index, index) {
                        // load the build side key
                        %currentBuildKey = memref.load %linkedListKey[%arg1] : memref<?xi32>
                        %cmp = arith.cmpi "eq", %probeKey, %currentBuildKey : i32

                        // if probeKey matches, increment the prefixSumArray sum
                        %newCount = scf.if %cmp -> (index){
                            %newCount = arith.addi %arg2, %oneIndex : index
                            scf.yield %newCount : index
                        }
                        else{
                            scf.yield %arg2 : index
                        }

                        %nextIndex = memref.load %linkedListnextIndex[%arg1] : memref<?xindex>
                        %condition = arith.cmpi "ne", %nextIndex, %negOneIndex : index
                        scf.condition(%condition) %nextIndex, %newCount : index, index
                        
                    } do {
                        ^bb0(%arg3: index, %arg4: index):
                            scf.yield %arg3, %arg4 : index, index
                    }
                    memref.store %finalCount, %sharedMemPrefixSum[%threadId] : memref<1024xindex, 3>

                }
                
                gpu.barrier // we need this so that all the warps are done computing the thread local sums

                //Step 2: Compute the global prefix sum

                %isThreadZero = arith.cmpi "eq", %threadId, %zeroIndex : index
                scf.if %isThreadZero {
                    
                    // threadBlockOffset stores the total number of elements needed by the entire thread block in the result array
                    %threadBlockOffset = arith.constant 0 : index

                    //For all threads in thread block
                    %totalOffset = scf.for %i = %zeroIndex to %blockDim step %oneIndex iter_args(%offset = %threadBlockOffset) -> index {

                        %currentThreadGlobalIndex = arith.addi %globalThreadOffsetInBlocks, %i : index
                        %isValid = arith.cmpi "ult", %currentThreadGlobalIndex, %probeRelationRows : index
                        %newOffset = scf.if %isValid -> (index){

                            %currentCount = memref.load %sharedMemPrefixSum[%i] : memref<1024xindex, 3>

                            //sharedMemPrefixSum[i] stores the starting index to write from for thread i, which is 0 for thread 0
                            memref.store %offset, %sharedMemPrefixSum[%i] : memref<1024xindex, 3>
                            %newOffset = arith.addi %offset, %currentCount : index
                            scf.yield %newOffset : index
                        }
                        else{
                            scf.yield %offset : index
                        }
                        scf.yield %newOffset : index

                    }
                    %totalOffsetI32 = arith.index_cast %totalOffset : index to i32
                    
                    %currentResultSizeI32 = memref.atomic_rmw addi %totalOffsetI32, %globalBlockOffset[%zeroIndex] : (i32, memref<1xi32>) -> i32
                    %currentResultSize = arith.index_cast %currentResultSizeI32 : i32 to index

                    memref.store %currentResultSize, %sharedMemTotalSum[%zeroIndex] : memref<1xindex, 3>

                }
                gpu.barrier

                // step 3: calculate global prefixSum for all threads
                %globalOffset = memref.load %sharedMemTotalSum[%zeroIndex] : memref<1xindex, 3>

                // add thread local prefixSum to global offset
                %threadBlockOffset = memref.load %sharedMemPrefixSum[%threadId] : memref<1024xindex, 3>
                %globalPrefixValue = arith.addi %threadBlockOffset, %globalOffset : index

                // store the global prefixSum
                memref.store %globalPrefixValue, %prefixSumArray[%globalThreadIndex] : memref<?xindex>
            }
            gpu.return
        }
    }

    gpu.module @kernelProbe {

        func.func @hash(%probeKey : i32, %hashTableSize : i32) -> i32{
            //%modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %hashTableSize : i32
            return %hashValue : i32
        }

        gpu.func @probe (%probeRelation : memref<?xi32>, %probeRelationRows : index, %hashTableSize : i32, %hashTablePointers : memref<?xi32>,
            %linkedListKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  
            %prefixSumArray : memref<?xindex>, %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>)
            private (%privateWriteIndex : memref<1xindex>)
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // globalThreadIndex = blockDim * blockId + threadId
            %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
            %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

            // check if the thread is valid
            %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %probeRelationRows : index

            scf.if %isThreadValid {
                //constants
                %negOneIndex = arith.constant -1 : index 
                %zeroIndex = arith.constant 0 : index
                %oneIndex = arith.constant 1 : index

                %negOneI32 = arith.constant -1 : i32
                
                // For each thread, compare its probeKey with all probeKeys in the relevant hash bucket chain

                %probeKey = memref.load %probeRelation[%globalThreadIndex] : memref<?xi32>

                %hashValueI32 = func.call @hash(%probeKey, %hashTableSize) : (i32, i32) -> i32
                %hashValue = arith.index_cast %hashValueI32 : i32 to index

                // To get the first node in the linked list
                %currentBucketIndexI32 = memref.load %hashTablePointers[%hashValue] : memref<?xi32>
                
                %checkBucketNotEmpty = arith.cmpi "ne", %currentBucketIndexI32, %negOneI32 : i32 

                //store privateWriteIndex in private variable
                %currentWriteIndex = memref.load %prefixSumArray[%globalThreadIndex] : memref<?xindex>
                memref.store %currentWriteIndex, %privateWriteIndex[%zeroIndex] : memref<1xindex>

                scf.if %checkBucketNotEmpty{
                    %currentBucketIndex = arith.index_cast %currentBucketIndexI32 : i32 to index
                    %globalThreadIndexI32 = arith.index_cast %globalThreadIndex : index to i32
                    
                    // do while loop to iterate over the linked list
                    %res = scf.while (%arg1 = %currentBucketIndex) :(index) -> index {
                        // load the current probeKey
                        %currentBuildKey = memref.load %linkedListKey[%arg1] : memref<?xi32>

                        // compare the probeKeys
                        %cmp = arith.cmpi "eq", %probeKey, %currentBuildKey : i32

                        // if probeKey matches, store the rowIDs and increment the current privateWriteIndex
                        scf.if %cmp {
                            %buildRelationRowID = memref.load %linkedListRowId[%arg1] : memref<?xindex>
                            %buildRelationRowIDI32 = arith.index_cast %buildRelationRowID : index to i32

                            %currentPrivateWriteIndex = memref.load %privateWriteIndex[%zeroIndex] : memref<1xindex>

                            // Store the build relation's rowID
                            memref.store %buildRelationRowIDI32, %resultIndicesR[%currentPrivateWriteIndex] : memref<?xi32>
                            // Store the probe relation's rowID
                            memref.store %globalThreadIndexI32, %resultIndicesS[%currentPrivateWriteIndex] : memref<?xi32>
                            
                            %newPrivateWriteIndex = arith.addi %currentPrivateWriteIndex, %oneIndex : index
                            memref.store %newPrivateWriteIndex, %privateWriteIndex[%zeroIndex] : memref<1xindex>
                        }
                        
                        // move to the next node in the linked list
                        %nextIndex = memref.load %linkedListnextIndex[%arg1] : memref<?xindex>
                        %condition = arith.cmpi "ne", %nextIndex, %negOneIndex : index
                        scf.condition(%condition) %nextIndex : index

                    } do {
                        ^bb0(%arg2: index):
                            scf.yield %arg2 : index
                    }
                }
            }
            gpu.return
        }
    }    
    
    func.func @main() {
        // Constants
        %zeroIndex = arith.constant 0 : index

        // Relation and Relation sizes are global variables
        %buildRelationRowsMemref = memref.get_global @buildRelationRows : memref<1xindex>
        %buildRelationRows = memref.load %buildRelationRowsMemref[%zeroIndex] : memref<1xindex>
        
        %probeRelationRowsMemref = memref.get_global @probeRelationRows : memref<1xindex>
        %probeRelationRows = memref.load %probeRelationRowsMemref[%zeroIndex] : memref<1xindex>

        // Hash table size
        %hashTableSizeMemref = memref.get_global @hashTableSize : memref<1xindex>
        %hashTableSize = memref.load %hashTableSizeMemref[%zeroIndex] : memref<1xindex>

        // Number of threads per block
        %numberOfThreadsPerBlockMemref = memref.get_global @numberOfThreadsPerBlock : memref<1xindex>
        %numberOfThreadsPerBlock = memref.load %numberOfThreadsPerBlockMemref[%zeroIndex] : memref<1xindex>

        // Allocate and initialize the memrefs of pkeys for both relations
        %hostBuildRelation = memref.alloc(%buildRelationRows) : memref<?xi32>
        call @initRelationR(%hostBuildRelation) : (memref<?xi32>) -> ()
        %hostProbeRelation = memref.alloc(%probeRelationRows) : memref<?xi32>
        call @initRelationS(%hostProbeRelation) : (memref<?xi32>) -> ()

        // %dst = memref.cast %hostBuildRelation : memref<?xi32> to memref<*xi32>
        // call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        // %dst1 = memref.cast %hostProbeRelation : memref<?xi32> to memref<*xi32>
        // call @printMemrefI32(%dst1) : (memref<*xi32>) -> ()

        // Allocate device memory for the relations
        %deviceBuildRelation = gpu.alloc(%buildRelationRows) : memref<?xi32>
        gpu.memcpy %deviceBuildRelation, %hostBuildRelation : memref<?xi32>, memref<?xi32>
        %deviceProbeRelation = gpu.alloc(%probeRelationRows) : memref<?xi32>
        gpu.memcpy %deviceProbeRelation, %hostProbeRelation : memref<?xi32>, memref<?xi32>

        // Number of rows in the build relation is the numTuples in the linked list
        // Allocate device memory for the hash table
        %linkedListKey, %linkedListRowId, %linkedListnextIndex, %hashTablePointers = func.call @allocateHashTable(%buildRelationRows, %hashTableSize) 
        : (index, index) -> (memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>)

        func.call @initializeHashTable(%hashTableSize, %hashTablePointers) : (index, memref<?xi32>) -> ()
        %hashTableSizeI32 = arith.index_cast %hashTableSize : index to i32

        func.call @buildTable(%deviceBuildRelation, %buildRelationRows, %hashTablePointers, %linkedListKey, %linkedListRowId, %linkedListnextIndex, %hashTableSizeI32) 
        : (memref<?xi32>, index, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, i32) -> ()

        // Print hash-table

        // %hashTablePointers_ = memref.alloc(%hashTableSize) : memref<?xi32>
        // gpu.memcpy %hashTablePointers_, %hashTablePointers : memref<?xi32>, memref<?xi32>
        // %d = memref.cast %hashTablePointers_ : memref<?xi32> to memref<*xi32>
        // call @printMemrefI32(%d) : (memref<*xi32>) -> ()

        // %h_linkedListRowId = memref.alloc(%linkedListRowId) : memref<?xindex>
        // gpu.memcpy %h_linkedListRowId, %prefixSumArray : memref<?xindex>, memref<?xindex>
        // %dst = memref.cast %h_linkedListRowId : memref<?xindex> to memref<*xindex>
        // call @printMemrefInd(%dst) : (memref<*xindex>) -> ()


        // Allocate memref for prefixSumArray sum array
        %prefixSumArray = gpu.alloc(%probeRelationRows) : memref<?xindex>

        // Get the resultSize from count phase
        %resultSize = func.call @countRows(%deviceProbeRelation, %probeRelationRows, %hashTablePointers, %linkedListKey, %linkedListRowId, %linkedListnextIndex, %prefixSumArray, %hashTableSizeI32)
        : (memref<?xi32>, index,  memref<?xi32>,  memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xindex>, i32) -> index

        
        // print result size
        %resultSizeI32 = arith.index_cast %resultSize : index to i32
        func.call @debugI32(%resultSizeI32) : (i32) -> ()

         // check if resultSize is not 0
        %checkBucketNotEmpty = arith.cmpi "ne", %resultSize, %zeroIndex : index
        scf.if %checkBucketNotEmpty {

            // Allocate device memory for the result
            %resultIndicesR = gpu.alloc(%resultSize) : memref<?xi32>
            %resultIndicesS = gpu.alloc(%resultSize) : memref<?xi32>
        
            func.call @probeRelation(%deviceProbeRelation, %probeRelationRows, %hashTableSizeI32, %hashTablePointers, %linkedListKey, %linkedListRowId, %linkedListnextIndex, %prefixSumArray, %resultIndicesR, %resultIndicesS)
            : (memref<?xi32>, index, i32, memref<?xi32>,  memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xi32>, memref<?xi32>) -> ()

            // transfer the result back to host
            %hostResultIndicesR = memref.alloc(%resultSize) : memref<?xi32>
            %hostResultIndicesS = memref.alloc(%resultSize) : memref<?xi32>

            gpu.memcpy %hostResultIndicesR, %resultIndicesR : memref<?xi32>, memref<?xi32>
            gpu.memcpy %hostResultIndicesS, %resultIndicesS : memref<?xi32>, memref<?xi32>
            
            // Print the result

            // %dstr = memref.cast %hostResultIndicesR : memref<?xi32> to memref<*xi32>
            // call @printMemrefI32(%dstr) : (memref<*xi32>) -> ()

            // %dsts = memref.cast %hostResultIndicesS : memref<?xi32> to memref<*xi32>
            // call @printMemrefI32(%dsts) : (memref<*xi32>) -> ()


            // check the result of join

            // %success = func.call @check(%hostBuildRelation, %hostProbeRelation, %hostResultIndicesR, %hostResultIndicesS)
            //  : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32

            // // print success
            // func.call @debugI32(%success) : (i32) -> ()
        
        }
        else{
            %hostResultIndicesR = memref.alloc(%resultSize) : memref<?xi32>
            %hostResultIndicesS = memref.alloc(%resultSize) : memref<?xi32>

            %success = func.call @check(%hostBuildRelation, %hostProbeRelation, %hostResultIndicesR, %hostResultIndicesS)
            : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32

            // print success
            func.call @debugI32(%success) : (i32) -> ()
        }

        // DO MEMREF DEALLOCATIONS
        
        return
    }

    func.func private @initRelationR(memref<?xi32>)
    func.func private @initRelationS(memref<?xi32>)
    func.func private @initRelationIndex(memref<?xi32>)
    func.func private @check(memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32
    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefInd(memref<*xindex>)
    func.func private @startTimer()
    func.func private @endTimer()
}