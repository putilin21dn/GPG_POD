#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>


__global__ void swap(double* mt, int row1, int row2, int n){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  int i;
  for(i = idx; i < n+1; i += offset){
    double tmp = mt[row1 + i*n];
    mt[row1 + i*n] = mt[row2 + i*n];
    mt[row2 + i*n] = tmp;
  }

}

__global__ void update(double* mt, int n, int k)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;
  for (int i = k + idx + 1; i < n; i += offsetx)
  {
    double factor = mt[i + k * n] / mt[k + k * n];
    for (int j = k + idy + 1; j < n + 1; j += offsety)
      mt[i + j * n] += -factor * mt[k + j * n];
  }
}

void print_matrix(double* h_A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n+1; j++) {
            std::cout << h_A[j * n + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}

struct Compare {
    __device__ bool operator()(double a, double b) const {
        return fabs(a) < fabs(b);
    }
};

void forward(double *A, int n) {
    std::vector<double> h_A(n * (n+1));
    Compare cmp;
    for (int k = 0; k < n; k++) { // номер строки
        thrust::device_ptr<const double> col = thrust::device_pointer_cast(&A[k * n + k]);
        auto max_iter = thrust::max_element(thrust::device, col, col + n - k, cmp);
        int max_row = max_iter - col + k;

        //swap
        if (max_row != k)
            swap <<<1024, 1024>>> (A, k, max_row, n);

        //update
        dim3 blockSize(32, 32);
        //dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
        dim3 gridSize(32,32);
        update<<<gridSize, blockSize>>>(A, n, k);
    }
}

void backward(const double *A,  double * x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = A[n*n + i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[j * n + i] * x[j];
        }
        x[i] /= A[i * n + i];
    }
}

int main() {
    int n;

    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    std::cin >> n;


    // Считываем матрицу и вектор b
    std::vector<double> A(n * (n+1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> A[j * n + i];  // Хранение матрицы по столбцам
        }
    }
    // матрица b
    for (int i = 0; i < n; i++){
      std::cin >> A[n*n + i];
    }

    // std::cerr << n << '\n';
    //  for (int i = 0; i < n; i++) {
    //      for (int j = 0; j < n+1; j++) {
    //          std::cerr << A[i + j *n] << ' ';
    //      }
    //      std::cerr << '\n';
    //  }

    double *d_A;
    cudaMalloc(&d_A, n * (n+1) * sizeof(double));
    cudaMemcpy(d_A, A.data(), n * (n+1) * sizeof(double), cudaMemcpyHostToDevice);


    // Прямой ход
    forward(d_A,  n);

    double *  A_g = (double *) malloc(n*(n+1) * sizeof(double));
    double * x  = (double *) malloc(n * sizeof(double));

    cudaMemcpy(A_g, d_A, n*(n+1) * sizeof(double), cudaMemcpyDeviceToHost);

    // Обратный ход
    backward(A_g, x, n);
    std::cout.precision(10);

    for (int i = 0; i < n; i++)
    {
      std::cout << x[i] << std::scientific  << " ";
    }

    cudaFree(d_A);
    return 0;
}