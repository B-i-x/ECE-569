__global__ void histo_kernel(unsigned int *buffer, long size, unsigned int *histo) {
    __shared__ unsigned int histo_private[24];

    // Initialize shared memory
    if (threadIdx.x < 24) {
        histo_private[threadIdx.x] = 0;
    }
    __syncthreads();

    // Compute histogram
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int k = i; k < size; k += blockDim.x * gridDim.x) {
        int position = buffer[k] % 24;
        atomicAdd(&histo_private[position], 1);
    }
    __syncthreads();

    // Write back to global memory
    if (threadIdx.x < 24) {
        atomicAdd(&histo[threadIdx.x], histo_private[threadIdx.x]);
    }
}