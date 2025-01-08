#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

using namespace std;
using uint = unsigned int;
using ll = long long;

const int full_size = 1 << 27;
const int concat_size = 1 << 5;
const int shared_size = 1 << 12;
const int blocks_cnt = full_size / shared_size;

__global__ void scan_part(uint* arr, uint* res, ll n)
{
  __shared__ uint arr_part[shared_size];
  ll l = blockIdx.x * shared_size;
  ll r = blockIdx.x * shared_size + shared_size;
  for (ll i = l + threadIdx.x; i < r; i += blockDim.x)
    arr_part[i - l] = (arr[i] >> n) & 1;
  __syncthreads();
  ll step = 2;
  while (step <= shared_size)
  {
    for (ll i = (threadIdx.x + 1) * step - 1; i < shared_size; i += blockDim.x * step)
      arr_part[i] += arr_part[i - step / 2];
    step <<= 1;
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    arr_part[shared_size - 1] = 0;
  __syncthreads();
  while (step > 1)
  {
    for (ll i = (threadIdx.x + 1) * step - 1; i < shared_size; i += blockDim.x * step)
    {
      uint tmp = arr_part[i];
      arr_part[i] += arr_part[i - step / 2];
      arr_part[i - step / 2] = tmp;
    }
    step >>= 1;
    __syncthreads();
  }
  __syncthreads();
  for (ll i = 1 + l + threadIdx.x; i < r; i += blockDim.x)
    res[i - 1] = arr_part[i - l];
  __syncthreads();
  if (threadIdx.x == 0)
    res[r - 1] = res[r - 2] + ((arr[r - 1] >> n) & 1);
  __syncthreads();
}

__global__ void scan_all(uint* scan_part, uint* res, ll k)
{
  __shared__ uint arr_part[concat_size];
  ll l = blockIdx.x * concat_size * k;
  ll r = min((ll)full_size, (blockIdx.x + 1) * concat_size * k);
  for (ll i = l + k - 1 + threadIdx.x * k; i < r; i += blockDim.x * k)
    arr_part[(i - l - k + 1) / k] = scan_part[i];
  __syncthreads();
  ll step = 2;
  while (step <= concat_size)
  {
    for (ll i = (threadIdx.x + 1) * step - 1; i < concat_size; i += blockDim.x * step)
      arr_part[i] += arr_part[i - step / 2];
    step <<= 1;
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    arr_part[concat_size - 1] = 0;
  __syncthreads();
  while (step > 1)
  {
    for (ll i = (threadIdx.x + 1) * step - 1; i < concat_size; i += blockDim.x * step)
    {
      uint tmp = arr_part[i];
      arr_part[i] += arr_part[i - step / 2];
      arr_part[i - step / 2] = tmp;
    }
    step >>= 1;
    __syncthreads();
  }
  __syncthreads();
  for (ll i = l + threadIdx.x; i < r; i += blockDim.x)
      res[i] = scan_part[i] + arr_part[(i - l) / k];
  __syncthreads();
}

__global__ void radix_sort(uint* arr, uint* scan, uint* res, ll n, ll k)
{
  ll idx = blockDim.x * blockIdx.x + threadIdx.x;
  ll offset = blockDim.x * gridDim.x;
  for (ll i = idx; i < n; i += offset)
  {
    ll bit = (arr[i] >> k) & 1;
    if (bit == 1)
      res[n - scan[n - 1] + scan[i] - 1] = arr[i];
    else
      res[i - scan[i]] = arr[i];
  }
}

void radix_sort(uint* arr, ll n)
{
  uint* dev_arr;
  cudaMalloc(&dev_arr, sizeof(uint) * full_size);
  cudaMemset(dev_arr, 0xFF, sizeof(uint) * full_size);
  cudaMemcpy(dev_arr, arr, sizeof(uint) * n, cudaMemcpyHostToDevice);
  uint* dev_tmp;
  uint* dev_scan;
  cudaMalloc(&dev_tmp, sizeof(uint) * full_size);
  cudaMalloc(&dev_scan, sizeof(uint) * full_size);
  for (ll i = 0; i < 32; i++)
  {
    scan_part <<<blocks_cnt, 1024>>> (dev_arr, dev_tmp, i);
    ll cur_len = shared_size;
    for (ll j = 0; j < 3; j++)
    {
      scan_all <<<blocks_cnt, 1024>>> (dev_tmp, dev_scan, cur_len);
      cur_len *= concat_size;
      swap(dev_scan, dev_tmp);
    }
    swap(dev_scan, dev_tmp);
    radix_sort <<<blocks_cnt, 1024>>> (dev_arr, dev_scan, dev_tmp, n, i);
    swap(dev_arr, dev_tmp);
  }
  //cerr << cudaGetErrorString(cudaGetLastError()) << ' ';
  cudaMemcpy(arr, dev_tmp, sizeof(uint) * n, cudaMemcpyDeviceToHost);
  //cerr << cudaGetErrorString(cudaGetLastError()) << '\n';
  cudaFree(dev_arr);
  cudaFree(dev_tmp);
  cudaFree(dev_scan);
}

int main()
{
  freopen(NULL, "rb", stdin);
  freopen(NULL, "wb", stdout);

  ll n;
  fread(&n, sizeof(uint), 1, stdin);
  cerr << n;
  uint* arr = (uint*)malloc(n * sizeof(uint));
  fread(arr, sizeof(uint), n, stdin); 
  radix_sort(arr, n);
  fwrite(arr, n, sizeof(uint), stdout);
  cerr << '\n' << cudaGetErrorString(cudaGetLastError());
  bool ok = true;
  for (int i = 1; i < n; i++)
    ok = ok && (arr[i - 1] <= arr[i]);
  cerr << "\n ok = " << ok;
}