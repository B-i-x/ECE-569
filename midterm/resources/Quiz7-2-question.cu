__global__ void my_kernel(float *X, float *Y, int InputSize) {
    __shared__ float XY[512];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }

    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
        __syncthreads();
        float in1 = XY[threadIdx.x - stride];
        __syncthreads();
        XY[threadIdx.x] += in1;
    }

    __syncthreads();

    if (i < InputSize) {
        Y[i] = XY[threadIdx.x];
    }
}

