module attributes {gpu.container_module} {

    func.func @main(){
        %0 = arith.constant 0 : index
        %1 = arith.constant 1 : index

        %y = memref.alloc(%0) : memref<?xi32>
        
        return
    }
}