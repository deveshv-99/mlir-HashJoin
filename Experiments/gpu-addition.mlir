// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/gpu-addition.mlir

func.func @main() {

    %size = arith.constant 4 : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %arg0 = memref.alloc(%size) : memref<?xf32>
    %arg1 = memref.alloc(%size) : memref<?xf32>
    %arg2 = memref.alloc(%size) : memref<?xf32>

    // Initialize arrays %arg0 and %arg1 here
    // Initialize the array with constant values
    %constant_1 = arith.constant 1.0 : f32
    %constant_2 = arith.constant 2.0 : f32


    // How to lower affine.for ???
     
    affine.for %i = 0 to %size {
        memref.store %constant_1, %arg0[%i] : memref<?xf32>
        memref.store %constant_2, %arg1[%i] : memref<?xf32>
    }
    %gpu_arg0 = gpu.alloc(%size) : memref<?xf32>
    // memref.store %constant_1, %arg0[%c0] : memref<4xf32>
    // memref.store %constant_1, %arg0[%c1] : memref<4xf32>
    // memref.store %constant_1, %arg0[%c2] : memref<4xf32>
    // memref.store %constant_1, %arg0[%c3] : memref<4xf32>

    // memref.store %constant_2, %arg1[%c0] : memref<4xf32>
    // memref.store %constant_2, %arg1[%c1] : memref<4xf32>
    // memref.store %constant_2, %arg1[%c2] : memref<4xf32>
    // memref.store %constant_2, %arg1[%c3] : memref<4xf32>

    %cast_arg0 = memref.cast %arg0 : memref<?xf32> to memref<*xf32>
    gpu.host_register %cast_arg0 : memref<*xf32>
    %cast_arg1 = memref.cast %arg1 : memref<?xf32> to memref<*xf32>
    gpu.host_register %cast_arg1 : memref<*xf32>
    %cast_arg2 = memref.cast %arg2 : memref<?xf32> to memref<*xf32>
    gpu.host_register %cast_arg2 : memref<*xf32>

    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1) 
        threads(%tx, %ty, %tz) in (%block_x = %size, %block_y = %c1, %block_z = %c1){
    
        // %val0 = memref.load %arg0[%bx, %tx] : memref<4xf32>
        // %val1 = memref.load %arg1[%bx, %tx] : memref<4xf32>

        %val0 = memref.load %arg0[%tx] : memref<?xf32>
        %val1 = memref.load %arg1[%tx] : memref<?xf32>

        %add = arith.addf %val0, %val1 : f32

        memref.store %add, %arg2[%tx] : memref<?xf32>

        gpu.terminator
    }

    call @printMemrefF32(%cast_arg2) : (memref<*xf32>) -> ()
    //CHECK: [3, 3, 3, 3]
    return 
}

func.func private @printMemrefF32(memref<*xf32>)