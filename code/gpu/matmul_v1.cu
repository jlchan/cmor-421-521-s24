#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

__global__ void matmul(int N, const float *A, const float *B, float *C) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < N) {
    float val = 0.f;
    for (int k = 0; k < N; ++k) {
      val += A[k + i * N] * B[j + k * N];
    }
    C[j + i * N] += val;    
  }
}

int main(int argc, char * argv[]){

  int N = 4096;
  if (argc > 1){
    N = atoi(argv[1]);
  }

  float * A = new float[N * N];
  float * B = new float[N * N];
  float * C = new float[N * N];

  for (int i = 0; i < N * N; ++i){
    A[i] = 0.f;
    B[i] = 0.f;
    C[i] = 0.f;
  }
  for (int i = 0; i < N; ++i){
    A[i + i * N] = 1.f; // identity
    B[i + i * N] = 1.f; // identity
  }

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_B;
  float * d_C;
  int size = N * N * sizeof(float);
  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);
  cudaMalloc((void **) &d_C, size);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

  // Next largest multiple of blockSize
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE; 
  printf("N = %d, numBlocks * blockSize = %d\n", N, numBlocks * BLOCKSIZE);
  dim3 gridDims(numBlocks, numBlocks);
  dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
  matmul <<< gridDims, blockDims >>> (N, d_A, d_B, d_C);

  // copy memory back to the CPU
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  
  float error = 0.f;
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      //      printf("C[%d,%d] = %f\n", i, j, C[j + i * N]);
      float Cij = 0.f;
      if (i==j){
	Cij = 1.f;
      }
      float diff = C[j + i * N] - Cij;
      error += fabs(diff);
    }
  }
  printf("error = %f\n", error);

  return 0;
}
