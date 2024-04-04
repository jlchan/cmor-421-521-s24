#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matvec(int N, const float *A, float *x, float *y){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N){
    float val = y[i];
    for (int j = 0; j < N; ++j){
      val += A[i + j * N] * x[j];  // coalesced
      //val += A[i + j * N] * x[j]; // non-coalesced
    }
    y[i] = val;
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
  float * x = new float[N];
  float * y = new float[N];

  for (int i = 0; i < N; ++i){
    A[i + i * N] = 1.f; // identity
    x[i] = 1.f;
    y[i] = 0.f;
  }

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_x;
  float * d_y;
  int size_A = N * N * sizeof(float);
  int size_x = N * sizeof(float);
  cudaMalloc((void **) &d_A, size_A);
  cudaMalloc((void **) &d_x, size_x);
  cudaMalloc((void **) &d_y, size_x);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size_x, cudaMemcpyHostToDevice);

  // Next largest multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize; 
  printf("N = %d, numBlocks * blockSize = %d\n", N, numBlocks * blockSize);

  matvec <<< numBlocks, blockSize >>> (N, d_A, d_x, d_y);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_x, cudaMemcpyDeviceToHost);
  
  float error = 0.f;
  for (int i = 0; i < N; ++i){
    error += fabs(y[i] - 1.f);
  }
  printf("error = %f\n", error);

#if 1
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < 10; ++i){
    matvec<<<numBlocks, blockSize>>>(N, d_A, d_x, d_y);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel: %6.2f ms \n", time);
  
#endif

  return 0;
}
