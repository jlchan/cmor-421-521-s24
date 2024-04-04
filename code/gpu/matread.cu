#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matread(int N, float *A){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (i < N && j < N) {
    //A[i + j * N] = 1.f; 
    A[j + i * N] = 1.f; 
  }
}

int main(int argc, char * argv[]){

  int N = 4096;
  int blockSize = 32;
  if (argc > 1){
    N = atoi(argv[1]);
    blockSize = atoi(argv[2]);
  }
  printf("N = %d, blockSize = %d\n", N, blockSize);

  float * A = new float[N * N];
  for (int i = 0; i < N * N; ++i){
    A[i] = 0.f; 
  }

  // allocate memory and copy to the GPU
  float * d_A;
  int size_A = N * N * sizeof(float);
  cudaMalloc((void **) &d_A, size_A);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

  // Next largest multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize; 
  printf("N = %d, numBlocks * blockSize = %d\n", N, numBlocks * blockSize);

  dim3 blockDims(blockSize, blockSize);
  dim3 gridDims(numBlocks, numBlocks);

#if 1
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < 10; ++i){
    matread <<< gridDims, blockDims >>> (N, d_A);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel: %6.2f ms.\n", time);
  
#endif

  return 0;
}
