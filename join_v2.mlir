gpu.module @kernelHashTable {

    gpu.func @initializeHashTable(%hashTableSize : index, %hashTablePointers : memref<?xi32>) kernel {
        %blockDim = gpu.block_dim x
        %blockId = gpu.block_id x
        %threadId = gpu.thread_id x

        // Global_thread_index = blockDim * blockId + threadId
        %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
        %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

        // Check if the thread is valid
        %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %hashTableSize : index

        scf.if %isThreadValid {
            %constantNeg1 = arith.constant -1 : i32
            memref.store %constantNeg1, %hashTablePointers[%globalThreadIndex] : memref<?xi32>
        }
        gpu.return 
    }
}

gpu.module @kernelBuild {

    func.func @hash(%key : i32) -> i32 {
        %modValue = arith.constant 10000 : i32
        %hashValue = arith.remui %key, %modValue : i32
        return %hashValue : i32
    }

    func.func @insertNodeInHashTable(%key : i32, %hashTablePointers : memref<?xi32>, %linkedListKeys : memref<?xi32>,
        %linkedListRowIds : memref<?xindex>, %linkedListNexts : memref<?xindex>, %freeIndex : memref<1xi32>, %globalThreadIndex : index) {

        %negOneIndex = arith.constant -1 : index
        %zeroIndex = arith.constant 0 : index 
        %oneInteger32 = arith.constant 1 : i32

        %indexInteger32 = memref.atomic_rmw addi %oneInteger32, %freeIndex[%zeroIndex] : (i32, memref<1xi32>) -> i32
        %index = arith.index_cast %indexInteger32 : i32 to index

        memref.store %key, %linkedListKeys[%index] : memref<?xi32>
        memref.store %globalThreadIndex, %linkedListRowIds[%index] : memref<?xindex>

        %hashValue = func.call @hash(%key) : (i32) -> i32
        %hashIndex = arith.index_cast %hashValue : i32 to index

        %oldIndexInteger32 = memref.atomic_rmw assign %indexInteger32, %hashTablePointers[%hashIndex] : (i32, memref<?xi32>) -> i32
        %oldIndex = arith.index_cast %oldIndexInteger32 : i32 to index
        memref.store %oldIndex, %linkedListNexts[%index] : memref<?xindex>

        return
    }

    gpu.func @build(%relation1 : memref<?xi32>, %relation1Rows : index,
            %hashTablePointers : memref<?xi32>, %linkedListKeys : memref<?xi32>, %linkedListRowIds : memref<?xindex>, 
            %linkedListNexts : memref<?xindex>, %freeIndex : memref<1xi32>)
        kernel {
        %blockDim = gpu.block_dim x
        %blockId = gpu.block_id x
        %threadId = gpu.thread_id x

        %globalThreadOffsetInBlocks = arith.muli %blockDim, %blockId : index
        %globalThreadIndex = arith.addi %globalThreadOffsetInBlocks, %threadId : index

        %isThreadValid = arith.cmpi "ult", %globalThreadIndex, %relation1Rows : index

        scf.if %isThreadValid {
            %key = memref.load %relation1[%globalThreadIndex] : memref<?xi32>
            func.call @insertNodeInHashTable(%key, %hashTablePointers, %linkedListKeys, %linkedListRowIds, %linkedListNexts, %freeIndex, %globalThreadIndex) 
             : (i32, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index) -> ()
        }

        gpu.return
    }
}