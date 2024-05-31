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

        %val = llvm.mlir.constant (43 : i32) : i32
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

        %value1 = llvm.load %ptr1 : !llvm.ptr<i32>
        %value2 = llvm.load %ptr2 : !llvm.ptr<i32>

        //call @debugI32(%value1) : (i32) -> ()
        //call @debugI32(%value2) : (i32) -> ()








        // Create a 2D array

        %x = llvm.alloca %ci32_1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr = llvm.getelementptr %x[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<array<2 x i32>>)
        //insert value at this pointer
        %ptptr = llvm.getelementptr %x[%c0, %c0] : (!llvm.ptr<array<2 x i32>> ,i32, i32) -> (!llvm.ptr<i32>)
        llvm.store %val, %ptptr : !llvm.ptr< i32>

        %valuee = llvm.load %ptptr : !llvm.ptr<i32>
        //call @debugI32(%valuee) : (i32) -> ()








        // final test

        %linked_list = llvm.alloca %ci32_1024 x !llvm.struct<(i32, i32, ptr<i32>)> :(i32) -> !llvm.ptr<struct<(i32, i32, ptr<i32>)>>
        %null_ptr = llvm.mlir.undef : !llvm.ptr<i32>

        %ptr3 = llvm.getelementptr %linked_list[%c0] : (!llvm.ptr<struct<(i32, i32, ptr<i32>)>>, i32) -> (!llvm.ptr<struct<(i32, i32, ptr<i32>)>>)

        %ptr_to_key = llvm.getelementptr %ptr3[%c0] : (!llvm.ptr<struct<(i32, i32, ptr<i32>)>>, i32) -> (!llvm.ptr<i32>)
        llvm.store %val, %ptr_to_key : !llvm.ptr<i32>

        %ptr_to_value = llvm.getelementptr %ptr3[%c1] : (!llvm.ptr<struct<(i32, i32, ptr<i32>)>>, i32) -> (!llvm.ptr<i32>)
        llvm.store %val1, %ptr_to_value : !llvm.ptr<i32>
        
        %ptr_to_ptr = llvm.getelementptr %ptr3[%c2] : (!llvm.ptr<struct<(i32, i32, ptr<i32>)>>, i32) -> (!llvm.ptr<ptr<i32>>)
        llvm.store %null_ptr, %ptr_to_ptr : !llvm.ptr<ptr<i32>>

        %key1 = llvm.load %ptr_to_key : !llvm.ptr<i32>
        //call @debugI32(%key1) : (i32) -> ()

        %value11 = llvm.load %ptr_to_value : !llvm.ptr<i32>
        //call @debugI32(%value11) : (i32) -> ()

        %ptr_to_ptr1 = llvm.load %ptr_to_ptr : !llvm.ptr<ptr<i32>>
        %valuee1 = llvm.load %ptr_to_ptr1 : !llvm.ptr<i32>
        // This will give an error as pointer is NULL
        //call @debugI32(%valuee1) : (i32) -> ()









        // sike.. now i need a self referential struct

        %ll = llvm.alloca %ci32_1024 x !llvm.struct<(i32, i32, ptr)> :(i32) -> !llvm.ptr<struct<(i32, i32, ptr)>>

        %ptr4 = llvm.getelementptr %ll[%c0] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<struct<(i32, i32, ptr)>>)
        %ptr_to_key1 = llvm.getelementptr %ptr4[%c0] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<i32>)
        llvm.store %val1, %ptr_to_key1 : !llvm.ptr<i32>

        %self_ptr = llvm.getelementptr %ptr4[%c2] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)
        //%self_ptr_val = llvm.load %self_ptr : !llvm.ptr<ptr>
        //builtin unrealized cast self_ptr to ptr
        %casted = llvm.bitcast %ptr4 : !llvm.ptr<struct<(i32, i32, ptr)>> to !llvm.ptr
        llvm.store %casted, %self_ptr : !llvm.ptr<ptr>

        //retrieve the value
        %ptr_to_ptr2 = llvm.load %self_ptr : !llvm.ptr<ptr>
        %retrieve =llvm.bitcast %ptr_to_ptr2 : !llvm.ptr to !llvm.ptr<struct<(i32, i32, ptr)>>
        %key_ptr = llvm.getelementptr %retrieve[%c0] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<i32>)
        %dd = llvm.load %key_ptr : !llvm.ptr<i32>
        //call @debugI32(%dd) : (i32) -> ()

        




        
        // using llvm.cmpxchg
        %xx = llvm.cmpxchg %ptr1, %val, %val1 "monotonic" "monotonic" : !llvm.ptr<i32>, i32
        %value = llvm.extractvalue %xx[0] : !llvm.struct<(i32, i1)>
        %success = llvm.extractvalue %xx[1] : !llvm.struct<(i32, i1)>
        call @debugI32(%value) : (i32) -> ()

        %valu1 = llvm.load %ptr1 : !llvm.ptr<i32>
        %valu2 = llvm.load %ptr2 : !llvm.ptr<i32>

        call @debugI32(%valu1) : (i32) -> ()
        call @debugI32(%valu2) : (i32) -> ()

        //convert success from i1 to i32
        %success_i32 = llvm.zext %success : i1 to i32
        call @debugI32(%success_i32) : (i32) -> ()








        // do while loop:

        %res = scf.while (%arg1 = %val) :(i32) -> i32 {
            // "Before" region.
            // In a "do-while" loop, this region contains the loop body.
            %vaalue = llvm.load %ptr1 : !llvm.ptr<i32>
            %x = llvm.cmpxchg %ptr1, %vaalue, %val1 "monotonic" "monotonic" : !llvm.ptr<i32>, i32
            %vvalue = llvm.extractvalue %x[0] : !llvm.struct<(i32, i1)>

            // condition
            %success1 = llvm.extractvalue %x[1] : !llvm.struct<(i32, i1)>

            // negate success
            %fail = llvm.xor %success1, %const_1 : i1

            scf.condition(%fail) %vvalue : i32

        } do {
            ^bb0(%arg2: i32):
            scf.yield %arg2 : i32
        }
        call @debugI32(%res) : (i32) -> ()
      

        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
}