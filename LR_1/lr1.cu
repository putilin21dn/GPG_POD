#include <stdio.h>
#include <cuda.h>

__global__ void subtraction(double* a, double* b, double* result, long long n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        result[idx] = a[idx] - b[idx];
        idx += offset;
    }

}


int main(){

  long long n,i;
  scanf("%llu", &n);

  double* a = (double*)malloc(n * sizeof(double));
  double* b = (double*)malloc(n * sizeof(double));
  double* result = (double*)malloc(n * sizeof(double));

  for(i=0; i<n; ++i){
    scanf("%lf",&a[i]);
  }

  for(i=0; i<n; ++i){
    scanf("%lf",&b[i]);
  }

  double *gpu_a, *gpu_b, *gpu_result;
  cudaMalloc(&gpu_a, n * sizeof(double));
  cudaMalloc(&gpu_b, n * sizeof(double));
  cudaMalloc(&gpu_result, n * sizeof(double));

  cudaMemcpy(gpu_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, b, n * sizeof(double), cudaMemcpyHostToDevice);


  subtraction<<<1024, 1024>>>(gpu_a, gpu_b, gpu_result, n);

  cudaMemcpy(result, gpu_result, n * sizeof(double), cudaMemcpyDeviceToHost);

  for(i=0; i<n; ++i){
    printf("%.10lf ", result[i]);
  }

  free(a);
  free(b);
  free(result);

  cudaFree(gpu_a);
  cudaFree(gpu_b);
  cudaFree(gpu_result);

  return 0;
}