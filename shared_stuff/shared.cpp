#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

std::chrono::high_resolution_clock::time_point start;

// For range of values in the key columns of the relation
int32_t lowerRange = 1;
int32_t upperRange = 10'000'000;


// Start the timer
extern "C" void startTimer(){
    start = std::chrono::high_resolution_clock::now();
}

// End the timer and print the time taken
extern "C" void endTimer(){
    auto stop = std::chrono::high_resolution_clock::now();

    static int x = 0;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "For " <<x;
    std::cout << ", time taken: " << duration.count() << " microseconds" << std::endl;
    x++;
}


// Initialize index as the key to the memref
extern "C" void initRelationIndex(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    //assign index values to alignedPtr
    for(auto i = 0; i < sizes; i++){
        alignedPtr[i] = (int32_t)i;
    }
}


// int generateRandomNumber(int min, int max) {
//     // Create a Mersenne Twister random number generator using the seed
//     std::mt19937 generator(seed);
//         // Use the high-resolution clock to get a new seed at each function call
//     unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

//     // Create a distribution in the desired range
//     std::uniform_int_distribution<int> distribution(min, max);

//     // Generate and return a random number in the specified range
//     return distribution(generator);
// }


// Initialize random values to the memref according to current time
extern "C" void initRelationR(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    // set seed for random number generation
    srand(time(0));
    
    //assign random integers to alignedPtr
    for(auto i = 0; i < sizes; i++){
        
        // Use rand() % range to generate numbers within a range 
        alignedPtr[i] = (int32_t)(rand() % (upperRange - lowerRange + 1) + lowerRange);
        // alignedPtr[i] = (int32_t)(generateRandomNumber(lowerRange, upperRange));
    }
}

// Initialize random values to the memref according to random_device
extern "C" void initRelationS(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    // set seed for random number generation
    std::random_device rd;
    srand(rd());
    
    //assign random integers to alignedPtr
    for(auto i = 0; i < sizes; i++){
        // Use rand() % range to generate numbers within a range 
        alignedPtr[i] = (int32_t)(rand() % (upperRange - lowerRange + 1) + lowerRange);
        // alignedPtr[i] = (int32_t)(generateRandomNumber(lowerRange, upperRange));
    }
}


/*
Takes input as 4 memrefs:

1. key column of table 1
2. key column of table 2
3. rowID column of result for table 1
4. rowID column of result for table 2

Returns 1 if the result generated is same as that received, 0 if it is not equal
*/
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

    // std::vector<int64_t> r(rAlignedPtr, rAlignedPtr + rSize);
    // std::vector<int64_t> s(sAlignedPtr, sAlignedPtr + sSize);

    // store result rowIDs values in inputs vector
    for(auto i = 0; i < result_size; i++){
        mlirResult[i] = std::make_pair(rIndicesMLIRAlignedPtr[i], sIndicesMLIRAlignedPtr[i]);
    }

    // Nested loop join
    int curr_index = 0;
    //std::cout<<"1: "<<result_size<<std::endl;
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
    //std::cout<<"2: "<<curr_index<<std::endl;
    // Sort the results lexicographically so they can be directly compared
    std::sort(result.begin(), result.end());
    std::sort(mlirResult.begin(), mlirResult.end());
    //std::cout<<"3: "<<result.size()<<std::endl;
    return result == mlirResult;
}