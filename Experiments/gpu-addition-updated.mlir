// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/gpu-addition.mlir

module attributes {gpu.container_module} {

    func.func @init(%arr: memref<?xf32>, %size: index) -> memref<?xf32> {

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %size step %c1 {
            %t = arith.index_cast %i : index to i32
            %r = arith.sitofp %t : i32 to f32
            memref.store %r, %arr[%i] : memref<?xf32>
        }
        return %arr: memref<?xf32>
    }

    
    gpu.module @kernels {
        gpu.func @add_arrays (%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel {
            %idx = gpu.thread_id x

            %val0 = memref.load %arg0[%idx] : memref<?xf32>
            %val1 = memref.load %arg1[%idx] : memref<?xf32>

            %result = arith.addf %val0, %val1 : f32
            memref.store %result, %arg2[%idx] : memref<?xf32>
            gpu.printf "Thread ID: %lld \t Result: %f\n" %idx, %result  : index, f32

            gpu.return
        }
    }

    func.func @main() {

        %size = arith.constant 4 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        

        %constant_1 = arith.constant 1.0 : f32
        %constant_2 = arith.constant 2.0 : f32

        %arg0_ = memref.alloc(%size) : memref<?xf32>
        %arg1_ = memref.alloc(%size) : memref<?xf32>
        %arg2 = memref.alloc(%size) : memref<?xf32>


        %arg0 = call @init(%arg0_, %size) : (memref<?xf32>, index) -> memref<?xf32>
        %arg1 = call @init(%arg1_, %size) : (memref<?xf32>, index) -> memref<?xf32>

        // affine.for %i = 0 to %size {
        //     memref.store %constant_1, %arg0[%i] : memref<?xf32>
        //     memref.store %constant_2, %arg1[%i] : memref<?xf32>
        // }

        %gpu_arg0 = gpu.alloc(%size) : memref<?xf32>
        %gpu_arg1 = gpu.alloc(%size) : memref<?xf32>
        %gpu_arg2 = gpu.alloc(%size) : memref<?xf32>

        gpu.memcpy %gpu_arg0, %arg0 : memref<?xf32>, memref<?xf32>
        gpu.memcpy %gpu_arg1, %arg1 : memref<?xf32>, memref<?xf32>

        
        
        gpu.launch_func @kernels::@add_arrays
            blocks in (%c1, %c1, %c1) 
            threads in (%size, %c1, %c1)
            args(%gpu_arg0 : memref<?xf32> , %gpu_arg1 : memref<?xf32>, %gpu_arg2 : memref<?xf32>)

        gpu.memcpy %arg2, %gpu_arg2 : memref<?xf32>, memref<?xf32>
        %printval = memref.cast %arg2 : memref<?xf32> to memref<*xf32>

        call @printMemrefF32(%printval) : (memref<*xf32>) -> ()
        //CHECK: [0, 2, 4, 8]

        memref.dealloc %arg0 : memref<?xf32>
        memref.dealloc %arg1 : memref<?xf32>
        memref.dealloc %arg2 : memref<?xf32>

        gpu.dealloc %gpu_arg0 : memref<?xf32>
        gpu.dealloc %gpu_arg1 : memref<?xf32>
        gpu.dealloc %gpu_arg2 : memref<?xf32>

        return
    }
    func.func private @printMemrefF32(memref<*xf32>)
}

