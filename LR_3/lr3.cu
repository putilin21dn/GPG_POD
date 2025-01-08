#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int MaxN=32;


using namespace std;



struct matrix3x3
{
    double v[3][3];

    __host__ __device__ void init()
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                v[i][j] = 0;
        }
    }

    __host__ __device__ double* operator[](int r)
    {
        return v[r];
    }
};

struct matrix3x1
{
    double v[3][1];

    __host__ __device__ void init()
    {
        for (int i = 0; i < 3; i++)
            v[i][0] = 0;
    }

    __host__ __device__ double& operator[](int r)
    {
        return v[r][0];
    }
};

struct matrix1x3
{
    double v[1][3];

    __host__ __device__ void init()
    {
        for (int i = 0; i < 3; i++)
            v[0][i] = 0;
    }

    __host__ __device__ double& operator[](int col)
    {
        return v[0][col];
    }
};

__constant__ matrix3x1 avg[MaxN];
__constant__ matrix3x3 cov[MaxN];


__host__ __device__ matrix3x3 operator*(double left_matrix, matrix3x3 right_matrix)
{
    matrix3x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            res[i][j] = left_matrix * double(right_matrix[i][j]);
    }
    return res;
}

__host__ __device__ matrix3x3 operator+(matrix3x3 left_matrix, matrix3x3 right_matrix)
{
    matrix3x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            res[i][j] = left_matrix[i][j] + right_matrix[i][j];
    }
    return res;
}

__host__ __device__ matrix3x1 operator*(double left_matrix, matrix3x1 right_matrix)
{
    matrix3x1 res;
    res.init();
    for (int i = 0; i < 3; i++)
        res[i] = left_matrix * right_matrix[i];
    return res;
}

__host__ __device__ matrix1x3 operator*(double left_matrix, matrix1x3 right_matrix)
{
    matrix1x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
        res[i] = left_matrix * right_matrix[i];
    return res;
}

__host__ __device__ matrix3x1 operator+(matrix3x1 left_matrix, matrix3x1 right_matrix)
{
    matrix3x1 res;
    res.init();
    for (int i = 0; i < 3; i++)
        res[i] = left_matrix[i] + right_matrix[i];
    return res;
}

__host__ __device__ matrix3x1 operator-(matrix3x1 left_matrix, matrix3x1 right_matrix)
{
    matrix3x1 res;
    res.init();
    for (int i = 0; i < 3; i++)
        res[i] = left_matrix[i] - right_matrix[i];
    return res;
}

__host__ __device__ matrix1x3 transposition(matrix3x1 a)
{
    matrix1x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
        res[i] = a[i];
    return res;
}

__host__ __device__ matrix3x3 operator*(matrix3x1 left_matrix, matrix1x3 right_matrix)
{
    matrix3x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            res[i][j] = left_matrix[i] * right_matrix[j];
    }
    return res;
}

__host__ __device__ matrix1x3 operator*(matrix1x3 left_matrix, matrix3x3 right_matrix)
{
    matrix1x3 res;
    res.init();
    for (int i = 0; i < 3; i++)
    {   
        for (int j = 0; j < 3; j++)
        {
          res[i] += left_matrix[j] * right_matrix[j][i];
        }
    }
    return res;
}

__host__ __device__ double operator*(matrix1x3 left_matrix, matrix3x1 right_matrix)
{
    double res = 0.;
    for (int i = 0; i < 3; i++)
    {
      res += left_matrix[i]*right_matrix[i];
    }

    return res;
}

__host__ __device__ double determinant(matrix3x3 mat)
{
    double term1 = mat[0][0] * mat[1][1] * mat[2][2];
    double term2 = mat[0][1] * mat[2][0] * mat[1][2];
    double term3 = mat[1][0] * mat[0][2] * mat[2][1];
    double term4 = -mat[2][0] * mat[1][1] * mat[0][2];
    double term5 = -mat[0][0] * mat[1][2] * mat[2][1];
    double term6 = -mat[2][2] * mat[1][0] * mat[0][1];
    return term1 + term2 + term3 + term4 + term5 + term6;
}

__host__ __device__ matrix3x3 inv(matrix3x3 mat) {
    double det = determinant(mat);
    matrix3x3 invMat;
    invMat.init();
    invMat[0][0] = mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2];
    invMat[0][1] = -(mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]);
    invMat[0][2] = mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2];
    invMat[1][0] = -(mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2]);
    invMat[1][1] = mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2];
    invMat[1][2] = -(mat[0][0] * mat[1][2] - mat[1][0] * mat[0][2]);
    invMat[2][0] = mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1];
    invMat[2][1] = -(mat[0][0] * mat[2][1] - mat[2][0] * mat[0][1]);
    invMat[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
    return (1.0 / det) * invMat;
}

__host__ __device__ double mahalanobis(uchar4 p, matrix3x1 avg, matrix3x3 cov)
{
    matrix3x1 points;
    points[0] = p.x;
    points[1] = p.y;
    points[2] = p.z;
    matrix3x1 tmp = points - avg;
    double num = (-1.) * ((transposition(tmp) * inv(cov)) * tmp);

    return num;
}

#define CSC(call, w, h)                                       \
do                                                      \
{											                                  \
	cudaError_t res = call;                               \
	if (res != cudaSuccess)                               \
  {                                                     \
		fprintf(stderr, "ERROR in %s:%d. Message: (%d, %d) %s\n",    \
				__FILE__, __LINE__, w, h, cudaGetErrorString(res));   \
	  exit(0);                                            \
	}                                                     \
} while(0)

__global__ void kernel(uchar4* data, int w, int h, int nc)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  int i,j;
  for (i = idx; i < w * h; i += offset)
  {
    uchar4 p = data[i];
    double maxim = -1e18;
    int class_new = -1;
    for (j = 0; j < nc; j++)
    {
      double dist = mahalanobis(p, avg[j], cov[j]);
      if (dist > maxim)
      {
        maxim = dist;
        class_new = j;
      }
    }
    p.w = class_new;
    data[i] = p;
  }
}


int main()
{
  char input_file[100], output_file[100];

  scanf("%s", input_file);
  scanf("%s", output_file);
  int nc,i,j, w, h;
  scanf("%d", &nc);

  vector <vector <int2>> classes(nc);
  for (i = 0; i < nc; i++)
  {
    int np;
    scanf("%d", &np);
    classes[i] = vector <int2>(np);
    for (j = 0; j < np; j++){
      scanf("%d %d", &classes[i][j].x, &classes[i][j].y);
    }

  }

  FILE *fp = fopen(input_file, "rb");
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);
  uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  matrix3x1 averages[32];
  matrix3x3 covariances[32];


  for (int idx = 0; idx < nc; idx++) {
    vector <matrix3x1> points(classes[idx].size());
    int np = points.size();
    for (int k = 0; k < np; k++) {
        uchar4 p = data[classes[idx][k].x + classes[idx][k].y * w];
        points[k][0] = p.x;
        points[k][1] = p.y;
        points[k][2] = p.z;
    }
    averages[idx].init();
    for (int k = 0; k < np; k++){
      averages[idx] = averages[idx] + points[k];
    }

    averages[idx] = (1.0 / np) * averages[idx];
    covariances[idx].init();
    for (int k = 0; k < np; k++) {
        matrix3x1 diff = points[k] - averages[idx];
        matrix3x3 tmp = diff * transposition(diff);
        covariances[idx] = covariances[idx] + tmp;
    }
    covariances[idx] = (1.0 / (np - 1)) * covariances[idx];
}

  cudaMemcpyToSymbol(avg, averages, sizeof(averages));
  cudaMemcpyToSymbol(cov, covariances, sizeof(covariances));

  uchar4* dev;
  CSC(cudaMalloc(&dev, sizeof(uchar4) * w * h), w, h);
  cudaMemcpy(dev, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice);

  kernel <<<1024, 1024>>> (dev, w, h, nc);
  CSC(cudaGetLastError(), w, h);
  uchar4* res_data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  CSC(cudaMemcpy(res_data, dev, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost), w, h);

  CSC(cudaFree(dev), w, h);

  fp = fopen(output_file, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(res_data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  free(data);
  free(res_data);
  return 0;
}
