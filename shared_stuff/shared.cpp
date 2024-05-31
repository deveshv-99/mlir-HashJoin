#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>


// Initialize index as the key to the memref
extern "C" void init_relation_index(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    //assign index values to alignedPtr
    for(auto i = 0; i < sizes; i++){
        alignedPtr[i] = (int32_t)i;
    }
}


// Initialize random values to the memref
extern "C" void init_relation(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    //assign random integers to alignedPtr
    for(auto i = 0; i < sizes; i++){
        alignedPtr[i] = (int32_t)(rand()%1000);
    }
}



// Check for the correctness of the result
extern "C" int32_t check(int32_t* basePtr1,int32_t* alignedPtr1, int64_t offset1, int64_t sizes1, int64_t strides1, 
    int32_t* basePtr2,int32_t* alignedPtr2, int64_t offset2, int64_t sizes2, int64_t strides2, 
    int32_t* r_basePtr,int32_t* r_alignedPtr, int64_t r_offset, int64_t r_sizes, int64_t r_strides,
    int32_t* s_basePtr,int32_t* s_alignedPtr, int64_t s_offset, int64_t s_sizes, int64_t s_strides){

    // std::cout <<"sizes " << r_sizes <<'\n';
    // std::cout <<"strides " << r_strides <<'\n';
    // std::cout <<"basePtr "<< r_basePtr;

    // allocate memory to store the memref values
    // int *temp = (int32_t *)malloc(sizes * sizeof(int32_t));
    // int *arr = new int(sizes); 
    
    // size of result memref
    int64_t result_size = r_sizes;

    std::vector< std::vector< int32_t > > inputs(result_size, std::vector< int32_t >(2, 0));
    std::vector< std::vector< int32_t > > result(result_size, std::vector< int32_t >(2, 0));

    int64_t input_size_r = sizes1;
    int64_t input_size_s = sizes2;
    std::vector< int64_t > r(input_size_r, 0);
    std::vector< int64_t > s(input_size_s, 0);

    // store input values in r and s vectors
    for(auto i = 0; i < input_size_r; i++){
        r[i] = alignedPtr1[i];
    }
    for(auto i = 0; i < input_size_s; i++){
        s[i] = alignedPtr2[i];
    }

    // store result rowIDs values in inputs vector
    for(auto i = 0; i < result_size; i++){
        inputs[i][0] = r_alignedPtr[i];
        inputs[i][1] = s_alignedPtr[i];
    }



    // int32_t *r_res = (int32_t *)malloc(result_size * sizeof(int32_t));
    // int32_t *s_res = (int32_t *)malloc(result_size * sizeof(int32_t));

    // Nested loop join
    int curr_index = 0;

    for(auto i = 0; i < sizes1; ++i){
        for(auto j = 0; j < sizes2; ++j){
            if(alignedPtr1[i] == alignedPtr2[j]){
                
                if(curr_index >= result_size){
                    return 0;
                }
                result[curr_index][0] = i;
                result[curr_index][1] = j;
                curr_index++;
            }
        }
    }


    if(curr_index != result_size){
        return 0;
    }
    if(curr_index == 0){
        return 1;
    }
    // Sort the result according to the first column
    std::sort(result.begin(), result.end());
    std::sort(inputs.begin(), inputs.end());
    

    // Check for the correctness of the result
    for(auto i = 0; i < result_size; i++){
        if(result[i][0] != inputs[i][0] || result[i][1] != inputs[i][1]){
            return 0;
        }
    }
    
    return 1;
}
