#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                                          \
do {                                                       \
    cudaError_t res = call;                                \
    if (res != cudaSuccess) {                              \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",   \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                           \
    }                                                      \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	  int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	  int offsety = blockDim.y * gridDim.y;
    int x, y, ip, jp;
    uchar4 p;
    for(y = idy; y < h; y += offsety)
		for(x = idx; x < w; x += offsetx) {
            double W[2][2];

            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 2; j++) {
                ip = max(min(x + i, w - 1), 0);  
                jp = max(min(y + j, h - 1), 0);
                p = tex2D< uchar4 >(tex, ip,jp);
                
                W[i][j] = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
              }
            }

            double Gx = W[1][1] - W[0][0] ;
            double Gy = W[1][0] - W[0][1] ;

            int gradLen = min(int(sqrt(Gx * Gx + Gy * Gy)), 255);
            
            out[y * w + x] = make_uchar4(gradLen, gradLen, gradLen, 0);
            
        }
}


int main() {
  char input_file[100], output_file[100];

  scanf("%s", input_file);
  scanf("%s", output_file);

  FILE *fp = fopen(input_file, "rb");

  int w, h;
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);
  uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  cudaArray* arr;
  cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
  CSC(cudaMallocArray(&arr, &ch, w, h));
  CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = arr;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = false;

  cudaTextureObject_t tex = 0;
  CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

  uchar4* dev_out;
  CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

  kernel <<<dim3(16, 16), dim3(32, 32)>>> (tex, dev_out, w, h);
  CSC(cudaGetLastError());

  uchar4* res_data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  CSC(cudaMemcpy(res_data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

  CSC(cudaDestroyTextureObject(tex));
  CSC(cudaFreeArray(arr));
  CSC(cudaFree(dev_out));

  fp = fopen(output_file, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(res_data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  free(data);
  free(res_data);
  return 0;
}

