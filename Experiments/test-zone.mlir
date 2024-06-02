     
module attributes {gpu.container_module} {
    memref.global constant @c : memref<1xindex> = dense<[1000000]>  
    func.func @test1(){
        %cidx0 = arith.constant 0 : index   
        %y = memref.get_global @c : memref<1xindex>
        %dst = memref.cast %y : memref<1xindex> to memref<*xindex>
        func.call @printMemrefInd(%dst) : (memref<*xindex>) -> ()
        return
    }
    
    func.func @main(){
        
        %0 = arith.constant 0 : index
        %1 = arith.constant 1 : index

        %y = memref.alloc(%0) : memref<?xi32>
        func.call @test1() : () -> ()
        
        return
    }
    func.func private @printMemrefInd(memref<*xindex>)
}
