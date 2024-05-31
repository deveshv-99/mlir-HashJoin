module {

func.func @scan_func(%arg0: !llvm.ptr<i8>) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c0_idx = arith.constant 0 : index

    // cast back the i8 pointer to a handle
    %handle = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>

    // get the first memref from the handle
    %first_ptr = llvm.getelementptr %handle[%c0] : (!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
    %first = llvm.load %first_ptr : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
    %first_memref = builtin.unrealized_conversion_cast %first : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>

    // get the second memref from the handle
    %second_ptr = llvm.getelementptr %handle[%c1] : (!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>
    %second = llvm.load %second_ptr : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>    
    %second_memref = builtin.unrealized_conversion_cast %second : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32>

    %first_val = memref.load %first_memref[%c0_idx] : memref<1xf32>
    %first_64 = arith.extf %first_val : f32 to f64
    %second_val = memref.load %second_memref[%c0_idx] : memref<1xi32>
    %second_64 = arith.extsi %second_val : i32 to i64
            
    %dst = memref.cast %first_memref: memref<1xf32> to memref<*xf32>
    func.call @printMemrefF32(%dst) : (memref<*xf32>) -> ()
    return
}

func.func @main() {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %col_count = llvm.mlir.constant(2 : i64) : i64
    %c0_idx = arith.constant 0 : index
    %f33 = llvm.mlir.constant(33.000000e+00 : f32) : f32
    %i42 = llvm.mlir.constant(42 : i32) : i32

    // create an array of structs (memrefs)
    %handle = llvm.alloca %col_count x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>

    // allocate memrefs
    %first_memref = memref.alloc() : memref<1xf32>
    %second_memref = memref.alloc() : memref<1xi32>

    // store values into memrefs
    memref.store %f33, %first_memref[%c0_idx] : memref<1xf32>
    memref.store %i42, %second_memref[%c0_idx] : memref<1xi32>

    // store the first memref into the array of memrefs
    %cast_first_memref = builtin.unrealized_conversion_cast %first_memref : memref<1xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %first = llvm.getelementptr %handle[%c0] : (!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>    
    llvm.store %cast_first_memref, %first : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>

    // store the second memref into the array of memrefs
    %cast_second_memref = builtin.unrealized_conversion_cast %second_memref : memref<1xi32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %second = llvm.getelementptr %handle[%c1] : (!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>    
    llvm.store %cast_second_memref, %second : !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>

    // cast the handle to a i8 pointer for scan_func
    %args = llvm.bitcast %handle: !llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    func.call @scan_func(%args) : (!llvm.ptr<i8>) -> ()    
    return
}

func.func private @printMemrefI32(memref<*xi32>)
func.func private @printMemrefF32(memref<*xf32>)

}