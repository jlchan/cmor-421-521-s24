#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void add(int N, const float *x, float *y){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < N){
    y[i] = y[i] + x[i];
  }
}

int main(void){

  int N = 1e6;
  float * x = new float[N];
  float * y = new float[N];

  for (int i = 0; i < N; ++i){
    x[i] = 1.f - i;
    y[i] = (float) i;
  }

  int size = N * sizeof(float);

  // allocate memory and copy to the GPU
  float * d_x;
  float * d_y;
  cudaMalloc((void **) &d_x, size);
  cudaMalloc((void **) &d_y, size);
  
  // copy memory over to the GPU
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  // call the add function  
  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
  
  float error = 0.f;
  for (int i = 0; i < N; ++i){
    error += fabs(y[i] - 1.f);
  }
  printf("error = %f\n", error);

  return 0;
}
