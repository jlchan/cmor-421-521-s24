#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define NCOLS 64

__global__ void matvec(int N, const float *A, float *x, float *y){
  
  __shared__ float s_x[NCOLS];
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int ii = threadIdx.x;
  while (ii < NCOLS){
    s_x[ii] = x[ii];
    ii += blockDim.x;
  }
  
  __syncthreads();
  
  if (i < N){
    float val = y[i];
    for (int j = 0; j < NCOLS; ++j){
      val += A[i + j * N] * s_x[j];  
      //val += A[i + j * N] * x[j];  
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

  float * A = new float[N * NCOLS];
  float * x = new float[NCOLS];
  float * y = new float[N];

  for (int i = 0; i < N; ++i){
    for (int j = 0; j < NCOLS; ++j){
      A[i + j * N] = 1.f / NCOLS;
    }
    if (i < NCOLS){
      x[i] = 1.f;
    }
    y[i] = 0.f;
  }

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_x;
  float * d_y;
  int size_A = N * NCOLS * sizeof(float);
  int size_x = NCOLS * sizeof(float);
  int size_y = N * sizeof(float);
  cudaMalloc((void **) &d_A, size_A);
  cudaMalloc((void **) &d_x, size_x);
  cudaMalloc((void **) &d_y, size_y);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);

  // Next largest multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize; 
  matvec <<< numBlocks, blockSize >>> (N, d_A, d_x, d_y);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);
  
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
    matvec <<< numBlocks, blockSize >>> (N, d_A, d_x, d_y);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel: %6.2f ms \n", time);
  
#endif

  return 0;
}
