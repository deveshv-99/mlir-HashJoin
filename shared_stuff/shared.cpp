#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cassert>

std::chrono::high_resolution_clock::time_point start;

// Timer start
extern "C" void start_timer(){
    start = std::chrono::high_resolution_clock::now();
}

// Timer end
extern "C" void end_timer(){
    auto stop = std::chrono::high_resolution_clock::now();

    static int x = 0;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "For " <<x;
    std::cout << ", time taken: " << duration.count() << " microseconds" << std::endl;
    x++;
}

// Initialize index as the key to the memref
extern "C" void init_relation_index(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    //assign index values to alignedPtr
    for(auto i = 0; i < sizes; i++){
        alignedPtr[i] = (int32_t)i;
    }
}

// Initialize random values to the memref
extern "C" void init_relation(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    // set seed for random number generation
    // srand(time(0));
    
    //assign random integers to alignedPtr
    for(auto i = 0; i < sizes; i++){
        alignedPtr[i] = (int32_t)(rand()%1000);
    }
}

// Check for the correctness of the result
extern "C" int32_t check(int32_t* rBasePtr, int32_t* rAlignedPtr, int64_t rOffset, int64_t rSize, int64_t rStride, 
    int32_t* sBasePtr, int32_t* sAlignedPtr, int64_t sOffset, int64_t sSize, int64_t sStride, 
    int32_t* rIndicesMLIRbasePtr, int32_t* rIndicesMLIRAlignedPtr, int64_t rIndicesMLIRoffset, int64_t rIndicesMLIRSize, int64_t rIndicesMLIRstride,
    int32_t* sIndicesMLIRBasePtr, int32_t* sIndicesMLIRAlignedPtr, int64_t sIndicesMLIROffset, int64_t sIndicesMLIRSize, int64_t sIndicesMLIRStride){
    
    assert (rIndicesMLIRSize == sIndicesMLIRSize);

    // size of result memref
    int64_t result_size = rIndicesMLIRSize;
    
    using ElemType = std::pair<int32_t, int32_t>;
    std::vector<ElemType> mlirResult(result_size);
    std::vector<ElemType> result(result_size);

    std::vector<int64_t> r(rAlignedPtr, rAlignedPtr + rSize);
    std::vector<int64_t> s(sAlignedPtr, sAlignedPtr + sSize);

    // store result rowIDs values in inputs vector
    for(auto i = 0; i < result_size; i++){
        mlirResult[i] = std::make_pair(rIndicesMLIRAlignedPtr[i], sIndicesMLIRAlignedPtr[i]);
    }

    // Nested loop join
    int curr_index = 0;

    for(auto i = 0; i < rSize; ++i){
        for(auto j = 0; j < sSize; ++j){
            if(rAlignedPtr[i] == sAlignedPtr[j]){
                
                if(curr_index >= result_size){
                    return -1;
                }
                result[curr_index] = std::make_pair(i, j);
                curr_index++;
            }
        }
    }

    // Sort the results lexicographically so they can be directly compared
    std::sort(result.begin(), result.end());
    std::sort(mlirResult.begin(), mlirResult.end());
    
    return result == mlirResult;
}
