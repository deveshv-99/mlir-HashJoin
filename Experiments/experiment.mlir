// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    func.func @debugI32(%val: i32) {
        %A = memref.alloc() : memref<i32>
        memref.store %val, %A[]: memref<i32>
        %U = memref.cast %A :  memref<i32> to memref<*xi32>

        func.call @printMemrefI32(%U): (memref<*xi32>) -> ()

        memref.dealloc %A : memref<i32>
        return
    }
    
    func.func @main() {

        // Constants
        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32
        %ci32_32 = arith.constant 32 : i32
        %ci32_1024 = arith.constant 1024 : i32

        %val = llvm.mlir.constant (45 : i32) : i32
        %val1 = llvm.mlir.constant (44 : i32) : i32
        %c0 = llvm.mlir.constant (0 : i32) : i32
        %c1 = llvm.mlir.constant (1 : i32) : i32
        %c2 = llvm.mlir.constant (2 : i32) : i32

        %const_1 = llvm.mlir.constant (1 : i1) : i1





        // use alloca to create array of 2 elements
        %a2 = llvm.alloca %ci32_2 x i32 : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr1 = llvm.getelementptr %a2[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)
        %ptr2 = llvm.getelementptr %a2[%c1] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)

        //insert value at this pointer
        llvm.store %val, %ptr1 : !llvm.ptr< i32>
        llvm.store %val1, %ptr2 : !llvm.ptr< i32>

        

        //call @debugI32(%dd) : (i32) -> ()

        
        // // using llvm.cmpxchg
        // %x = llvm.cmpxchg %ptr1, %val, %val1 "monotonic" "monotonic" : !llvm.ptr<i32>, i32
        // %value = llvm.extractvalue %x[0] : !llvm.struct<(i32, i1)>
        // %success = llvm.extractvalue %x[1] : !llvm.struct<(i32, i1)>
        // call @debugI32(%value) : (i32) -> ()

        // %value1 = llvm.load %ptr1 : !llvm.ptr<i32>
        // %value2 = llvm.load %ptr2 : !llvm.ptr<i32>

        // call @debugI32(%value1) : (i32) -> ()
        // call @debugI32(%value2) : (i32) -> ()

        // //convert success from i1 to i32
        // %success_i32 = llvm.zext %success : i1 to i32
        // call @debugI32(%success_i32) : (i32) -> ()





        // do while loop:

        %res = scf.while (%arg1 = %val) :(i32) -> i32 {
            // "Before" region.
            // In a "do-while" loop, this region contains the loop body.
            %vaalue = llvm.load %ptr1 : !llvm.ptr<i32>
            %x = llvm.cmpxchg %ptr1, %vaalue, %val1 "monotonic" "monotonic" : !llvm.ptr<i32>, i32
            %value = llvm.extractvalue %x[0] : !llvm.struct<(i32, i1)>

            // condition
            %success = llvm.extractvalue %x[1] : !llvm.struct<(i32, i1)>

            // negate success
            %fail = llvm.xor %success, %const_1 : i1

            scf.condition(%fail) %value : i32

        } do {
            ^bb0(%arg2: i32):
            scf.yield %arg2 : i32
        }
        call @debugI32(%res) : (i32) -> ()


        // pointer to a memref?
        %h_t= memref.alloc() : memref<2xi32>
        %ht1 = llvm.getelementptr %h_t[%c0] : (memref<2xi32>, i32) -> !llvm.ptr<i32>

        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
}