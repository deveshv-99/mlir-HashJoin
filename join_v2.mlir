// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

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
        %linkedListprobeKey = gpu.alloc(%numTuples) : memref<?xi32>
        %linkedListRowId = gpu.alloc(%numTuples) : memref<?xindex>
        %linkedListnextIndex = gpu.alloc(%numTuples) : memref<?xindex>

        // Allocate hash table
        %hashTablePointers = gpu.alloc(%hashTableSize) : memref<?xi32>
        
        // Return all of the allocated memory
        return %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex, %hashTablePointers
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
        %oneIndex = arith.constant 1 : index

        // //-------------> Keep threads per block constant at 256.. Need to change this?
        %threadsPerBlock = arith.constant 256 : index

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%hashTableSize, %threadsPerBlock) : (index, index) -> index

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelHashTable::@initializeHashTable
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex)
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%hashTableSize : index, %hashTablePointers : memref<?xi32>)

        func.call @endTimer() : () -> ()

        return
    }

    func.func @buildTable(%hostbuildRelation : memref<?xi32>, %buildRelationRows : index, %hashTablePointers : memref<?xi32>,
        %linkedListprobeKey : memref<?xi32>, %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>) {
        
        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index
        %zeroI32 = arith.constant 0 : i32

        //global variable for nextIndex free index in linked_list
        %freeIndex = gpu.alloc() : memref<1xi32>

        //store 0 initially
        gpu.memset %freeIndex, %zeroI32 : memref<1xi32>, i32

        // //-------------> Keep threads per block constant at 256.. Need to change this?
        %threadsPerBlock = arith.constant 256 : index

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%buildRelationRows, %threadsPerBlock) : (index, index) -> index

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelBuild::@build
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%hostbuildRelation : memref<?xi32>, %buildRelationRows : index, %hashTablePointers : memref<?xi32>, %linkedListprobeKey : memref<?xi32>,
            %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>)

        func.call @endTimer() : () -> ()
        return

    }

    func.func @countRows(%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,%linkedListprobeKey : memref<?xi32>,
        %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  %prefixSumArray : memref<?xindex>) -> (index) {

        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index

        %zeroI32 = arith.constant 0 : i32
        
        // //-------------> Keep threads per block constant at 256.. Need to change this?
        %threadsPerBlock = arith.constant 256 : index

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%probeRelationRows, %threadsPerBlock) : (index, index) -> index

        // Global variable for block offset
        %hostGlobalBlockOffset = memref.alloc() : memref<1xi32>
        memref.store %zeroI32, %hostGlobalBlockOffset[%zeroIndex] : memref<1xi32>

        %globalBlockOffset = gpu.alloc() : memref<1xi32>
        gpu.memcpy %globalBlockOffset, %hostGlobalBlockOffset : memref<1xi32>, memref<1xi32>

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelCount::@count 
        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListprobeKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,
            %prefixSumArray : memref<?xindex>, %globalBlockOffset : memref<1xi32>)

        func.call @endTimer() : () -> ()
        
        // After the kernel has executed, the globalBlockOffset contains the size of the result array
        gpu.memcpy %hostGlobalBlockOffset, %globalBlockOffset : memref<1xi32>, memref<1xi32>
        
        %resultSize = memref.load %hostGlobalBlockOffset[%zeroIndex] : memref<1xi32>
        %resultSizeIndex = arith.index_cast %resultSize : i32 to index

        return %resultSizeIndex : index
    }

    func.func @hostProbeRelation(%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>, 
        %linkedListprobeKey : memref<?xi32>, %linkedListRowId : memref<?xindex> , %linkedListnextIndex : memref<?xindex>, %prefixSumArray : memref<?xindex>,
        %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>){
        
        // Constants
        %zeroIndex = arith.constant 0 : index
        %oneIndex = arith.constant 1 : index

        %zeroI32 = arith.constant 0 : i32
        
        // //-------------> Keep threads per block constant at 256.. Need to change this?
        %threadsPerBlock = arith.constant 256 : index

        %numberOfBlocks = func.call @calculateNumberOfBlocks(%probeRelationRows, %threadsPerBlock) : (index, index) -> index

        func.call @startTimer() : () -> ()

        gpu.launch_func @kernelProbe::@probe

        blocks in (%numberOfBlocks, %oneIndex, %oneIndex) 
        threads in (%threadsPerBlock, %oneIndex, %oneIndex)
        args(%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListprobeKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,
            %prefixSumArray : memref<?xindex>, %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>)

        func.call @endTimer() : () -> ()
        return
    }
    
    gpu.module @kernelHashTable {

        gpu.func @initializeHashTable(%hashTableSize : index, %hashTablePointers : memref<?xi32>) 
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // Global_thread_index = blockDim * blockId + threadId
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

        func.func @hash(%probeKey : i32) -> i32{
            // return modulo 10000
            %modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %modValue : i32
            return %hashValue : i32
        }


        func.func @insertNodeInHashTable(%probeKey : i32, %hashTablePointers : memref<?xi32>, %linkedListprobeKey : memref<?xi32>,
            %linkedListRowId : memref<?xindex>, %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>, %globalThreadIndex : index) {

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

            // Insert probeKey and rowID into the new node
            memref.store %probeKey, %linkedListprobeKey[%index] : memref<?xi32>
            memref.store %globalThreadIndex, %linkedListRowId[%index] : memref<?xindex>

            // compute the hash value
            %hashValueI32 = func.call @hash(%probeKey) : (i32) -> i32
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

        gpu.func @build(%hostbuildRelation : memref<?xi32>, %buildRelationRows : index,
                %hashTablePointers : memref<?xi32>, %linkedListprobeKey : memref<?xi32>, %linkedListRowId : memref<?xindex>, 
                %linkedListnextIndex : memref<?xindex>, %freeIndex : memref<1xi32>)
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // Global_thread_index = blockDim * blockId + threadId
            %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
            %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

            // checkBucketNotEmpty if the thread is valid
            %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %buildRelationRows : index

            scf.if %isThreadValid {

                %probeKey = memref.load %hostbuildRelation[%globalThreadIndex] : memref<?xi32>
                
                func.call @insertNodeInHashTable(%probeKey, %hashTablePointers, %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex, %freeIndex, %globalThreadIndex) 
                : (i32, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index) -> ()
               
            }
            
            gpu.return
        }
    }

    gpu.module @kernelCount {

        func.func @hash(%probeKey : i32) -> i32{
            // return modulo 10000
            %modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %modValue : i32
            return %hashValue : i32
        }

        gpu.func @count (%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListprobeKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  
            %prefixSumArray : memref<?xindex>, %globalBlockOffset : memref<1xi32>)

            workgroup(%sharedMemPrefixSum : memref<256xindex, 3>, %sharedMemTotalSum : memref<1xindex, 3>)
            private(%privateVar: memref<1xindex>)
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // Global_thread_index = blockDim * blockId + threadId
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
                memref.store %zeroIndex, %sharedMemPrefixSum[%threadId] : memref<256xindex, 3>

                // Step 1: Compute the prefixSumArray sum for each thread, which is used for calculating the start index in the result array 
                // For each thread, compare its probeKey with all probeKeys in the relevant hash bucket chain

                %probeKey = memref.load %hostProbeRelation[%globalThreadIndex] : memref<?xi32>

                %hashValueI32 = func.call @hash(%probeKey) : (i32) -> i32
                %hashValue = arith.index_cast %hashValueI32 : i32 to index

                // To get the first node in the linked list
                %currentBucketIndexI32 = memref.load %hashTablePointers[%hashValue] : memref<?xi32>
                
                // check if the list is not empty
                %checkBucketNotEmpty = arith.cmpi "ne", %currentBucketIndexI32, %negOneI32 : i32 

                %checkBucketNotEmptyI32 = arith.index_cast %checkBucketNotEmpty : i1 to index

                scf.if %checkBucketNotEmpty{
                    %currentBucketIndex = arith.index_cast %currentBucketIndexI32 : i32 to index

                    %count = arith.constant 0 : index
                    // do while loop to iterate over the linked list
                    %res, %bucket_size = scf.while (%arg1 = %currentBucketIndex, %arg2 = %count) :(index, index) -> (index, index) {
                        // load the current probeKey
                        %currentBuildKey = memref.load %linkedListprobeKey[%arg1] : memref<?xi32>
                        // gpu.printf " %d  %d  \n" %hashValueI32 , %currentBuildKey : i32, i32
                        // compare the probeKeys
                        %cmp = arith.cmpi "eq", %probeKey, %currentBuildKey : i32

                        // if probeKey matches, increment the prefixSumArray sum
                        scf.if %cmp {
                            %currentCount = memref.load %sharedMemPrefixSum[%threadId] : memref<256xindex, 3>
                            %newCount = arith.addi %currentCount, %oneIndex : index
                            memref.store %newCount, %sharedMemPrefixSum[%threadId] : memref<256xindex, 3>
                        }
                        
                        // move to the next node in the linked list
                        %nextIndex = memref.load %linkedListnextIndex[%arg1] : memref<?xindex>

                        %condition = arith.cmpi "ne", %nextIndex, %negOneIndex : index
                        // Forward the argument (as result or "after" region argument).
                        %newCount = arith.addi %arg2, %oneIndex : index
                        scf.condition(%condition) %nextIndex, %newCount : index, index

                    } do {
                        ^bb0(%arg3: index, %arg4: index):
                            scf.yield %arg3, %arg4 : index, index
                    }

                }
                
                gpu.barrier // we need this so that all the warps are done computing the thread local sums

                //Step 2: Compute the global prefix sum

                %isThreadZero = arith.cmpi "eq", %threadId, %zeroIndex : index
                scf.if %isThreadZero {
                    
                    // %privateVar stores the current value of prefixSumArray sum
                    memref.store %zeroIndex, %privateVar[%zeroIndex] : memref<1xindex> 

                    //For all threads in thread block
                    scf.for %i = %zeroIndex to %blockDim step %oneIndex {

                        %currentThreadGlobalIndex = arith.addi %globalThreadOffsetInBlocks, %i : index
                        %isValid = arith.cmpi "ult", %currentThreadGlobalIndex, %probeRelationRows : index
                        scf.if %isValid{

                            %currentCount = memref.load %sharedMemPrefixSum[%i] : memref<256xindex, 3>
                            %currentBucketIndex = memref.load %privateVar[%zeroIndex] : memref<1xindex>
                            %newIndex = arith.addi %currentBucketIndex, %currentCount : index

                            //sharedMemPrefixSum[i] stores the starting index to write from for thread i, which is 0 for thread 0
                            memref.store %currentBucketIndex, %sharedMemPrefixSum[%i] : memref<256xindex, 3>
                            memref.store %newIndex, %privateVar[%zeroIndex] : memref<1xindex>
                        }
                    }

                    //Compute global block offset
                    %totalElements = memref.load %privateVar[%zeroIndex] : memref<1xindex>
                    %totalElementsI32 = arith.index_cast %totalElements : index to i32
                    
                    %currentResultSizeI32 = memref.atomic_rmw addi %totalElementsI32, %globalBlockOffset[%zeroIndex] : (i32, memref<1xi32>) -> i32
                    %currentResultSize = arith.index_cast %currentResultSizeI32 : i32 to index

                    memref.store %currentResultSize, %sharedMemTotalSum[%zeroIndex] : memref<1xindex, 3>

                }

                gpu.barrier

                // step 3: calculate global prefixSum for all threads
                %globalOffset = memref.load %sharedMemTotalSum[%zeroIndex] : memref<1xindex, 3>
                %globalOffsetI32 = arith.index_cast %globalOffset : index to i32

                // add thread local prefixSum to global offset
                %threadBlockOffset = memref.load %sharedMemPrefixSum[%threadId] : memref<256xindex, 3>
                %globalPrefixValue = arith.addi %threadBlockOffset, %globalOffset : index

                // store the global prefixSum
                memref.store %globalPrefixValue, %prefixSumArray[%globalThreadIndex] : memref<?xindex>

            }

            gpu.return
        }
    }

    gpu.module @kernelProbe {

        func.func @hash(%probeKey : i32) -> i32{
            // return modulo 10000
            %modValue = arith.constant 10000 : i32
            %hashValue = arith.remui %probeKey, %modValue : i32
            return %hashValue : i32
        }

        gpu.func @probe (%hostProbeRelation : memref<?xi32>, %probeRelationRows : index, %hashTablePointers : memref<?xi32>,
            %linkedListprobeKey : memref<?xi32>, %linkedListRowId  : memref<?xindex>, %linkedListnextIndex : memref<?xindex>,  
            %prefixSumArray : memref<?xindex>, %resultIndicesR : memref<?xi32>, %resultIndicesS : memref<?xi32>)
            private (%privateWriteIndex : memref<1xindex>)
            kernel
        {
            %blockDim = gpu.block_dim x
            %blockId = gpu.block_id x
            %threadId = gpu.thread_id x

            // Global_thread_index = blockDim * blockId + threadId
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

                %probeKey = memref.load %hostProbeRelation[%globalThreadIndex] : memref<?xi32>

                %hashValueI32 = func.call @hash(%probeKey) : (i32) -> i32
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
                        %currentBuildKey = memref.load %linkedListprobeKey[%arg1] : memref<?xi32>

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

        // Relation and Relation sizes have to be passed as arguments for now
        %buildRelationRows = arith.constant 100000 : index
        %probeRelationRows = arith.constant 100000 : index

        // Hash table size
        %hashTableSize = arith.constant 10000 : index

        // Allocate and initialize the memrefs of probeKeys for both relations
        %hostbuildRelation = memref.alloc(%buildRelationRows) : memref<?xi32>
        call @initRelation(%hostbuildRelation) : (memref<?xi32>) -> ()

        %hostProbeRelation = memref.alloc(%probeRelationRows) : memref<?xi32>
        call @initRelation(%hostProbeRelation) : (memref<?xi32>) -> ()

        // Allocate device memory for the relations
        %devicebuildRelation = gpu.alloc(%buildRelationRows) : memref<?xi32>
        gpu.memcpy %devicebuildRelation, %hostbuildRelation : memref<?xi32>, memref<?xi32>

        %deviceProbeRelation = gpu.alloc(%probeRelationRows) : memref<?xi32>
        gpu.memcpy %deviceProbeRelation, %hostProbeRelation : memref<?xi32>, memref<?xi32>

        // Number of rows in the build relation is the numTuples in the linked list
        // Allocate device memory for the hash table
        %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex, %hashTablePointers = func.call @allocateHashTable(%buildRelationRows, %hashTableSize) 
        : (index, index) -> (memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>)

        func.call @initializeHashTable(%hashTableSize, %hashTablePointers) : (index, memref<?xi32>) -> ()

        func.call @buildTable(%devicebuildRelation, %buildRelationRows, %hashTablePointers, %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex) 
        : (memref<?xi32>, index, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>) -> ()

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
        %resultSize = func.call @countRows(%deviceProbeRelation, %probeRelationRows, %hashTablePointers, %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex, %prefixSumArray)
        : (memref<?xi32>, index,  memref<?xi32>,  memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> index

        
        // print result size
        %resultSizeI32 = arith.index_cast %resultSize : index to i32
        func.call @debugI32(%resultSizeI32) : (i32) -> ()

         // check if resultSize is not 0
        %checkBucketNotEmpty = arith.cmpi "ne", %resultSize, %zeroIndex : index
        scf.if %checkBucketNotEmpty {

            // Allocate device memory for the result
            %resultIndicesR = gpu.alloc(%resultSize) : memref<?xi32>
            %resultIndicesS = gpu.alloc(%resultSize) : memref<?xi32>
        
            func.call @hostProbeRelation(%deviceProbeRelation, %probeRelationRows, %hashTablePointers, %linkedListprobeKey, %linkedListRowId, %linkedListnextIndex, %prefixSumArray, %resultIndicesR, %resultIndicesS)
            : (memref<?xi32>, index,  memref<?xi32>,  memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xi32>, memref<?xi32>) -> ()

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


            // checkBucketNotEmpty the result of join

            %success = func.call @check(%hostbuildRelation, %hostProbeRelation, %hostResultIndicesR, %hostResultIndicesS)
             : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32

            // print success
            func.call @debugI32(%success) : (i32) -> ()
        
        }
        else{
            %hostResultIndicesR = memref.alloc(%resultSize) : memref<?xi32>
            %hostResultIndicesS = memref.alloc(%resultSize) : memref<?xi32>

            // %success = func.call @check(%hostbuildRelation, %hostProbeRelation, %hostResultIndicesR, %hostResultIndicesS)
            // : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32

            // // print success
            // func.call @debugI32(%success) : (i32) -> ()
        }
        
        return
    }

    func.func private @initRelation(memref<?xi32>)
    func.func private @initRelationIndex(memref<?xi32>)
    func.func private @check(memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32
    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefInd(memref<*xindex>)
    func.func private @startTimer()
    func.func private @endTimer()
}