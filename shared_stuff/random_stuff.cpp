#include <iostream>

int memref(int* basePtr,int* alignedPtr, long offset, long sizes[], long strides[]){
  // Print the data
    long rows = sizes[0];
    long cols = sizes[1];

    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            std::cout << basePtr[i * strides[0] + j * strides[1]] << " ";
        }
        std::cout << std::endl;
    }
}


int main(){

    return 0;
}










// int main() {
//   std::cout << sizeof(long) << std::endl;
//   return 0;
// }