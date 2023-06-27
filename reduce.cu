#include <cuda.h>
#include <cuda_runtime.h>

// 相邻配对
__global__ void reduceNeighbored(int *idata, int *odata, size_t n) {
    size_t tid = threadIdx.x;

    int *data = idata + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)   odata[blockIdx.x] = data[0];
}

// 相邻配对 + 修改每个线程访问的数组元素
__global__ void reduceNeighbored(int *idata, int *odata, size_t n) {
    size_t tid = threadIdx.x;

    int *data = idata + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x>) {
            data[index] += data[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)   odata[blockIdx.x] = data[0];
}

// 交错配对
__global__ void reduceInterleaved(int *idata, int *odata, size_t n) {
    size_t tid = threadIdx.x;
    size_t idx = tid + blockDim.x * blockIdx.x;

    int *data = idata + blockDim.x * blockIdx.x;

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)  {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)  odata[blockIdx.x] = data[0];
}

// 展开循环   half blocks   grid.x / 2
__global__ void reduceUnrolling2(int *idata, int *odata, size_t n) {
    size_t tid = threadIdx.x;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x * 2;

    int *data = idata + blockDim.x * blockIdx.x * 2;
    if ((idx + blockDim.x) < n) data[tid] += data[tid + blockDim.x];
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)  odata[blockIdx.x] = data[0];
}


// 展开循环 & 线程束   half blocks   grid.x / 8
__global__ void reduceUnrollWarp8(int *idata, int *odata, size_t n) {
    size_t tid = threadIdx.x;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x * 8;

    int *data = idata + blockDim.x * blockIdx.x * 8;
    if ((idx + 7 * blockDim.x) < n) {
        int a0 = data[tid];
        int a1 = data[tid + blockDim.x];
        int a2 = data[tid + 2*blockDim.x];
        int a3 = data[tid + 3*blockDim.x];
        int a4 = data[tid + 4*blockDim.x];
        int a5 = data[tid + 5*blockDim.x];
        int a6 = data[tid + 6*blockDim.x];
        int a7 = data[tid + 7*blockDim.x];
        data[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *vmem = data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)  odata[blockIdx.x] = data[0];
}


int main(int argc, char **argv) {
    int dev = 0;
    cudaSetDevice(dev);

    bool bResult = false;

    int size = 1<<24;
    int blockSize = 512;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }
    dim3 block(blockSize, 1);
    dim3 grid((size + blockSize - 1) / blockSize);

    int *h_idata = (int *)malloc(size * sizeof(int));
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, size * sizeof(int));
    size_t istart, ielaps;
    int gpu_sum = 0;

    int *d_idata = NULL, *d_odata = NULL;
    cudaMalloc((void **)&d_idata, size * sizeof(int));
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    istart = second();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    elaps = second() - istart;
}