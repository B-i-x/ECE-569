unsigned int t = threadIdx.x;
unsigned int start = 2 * blockIdx.x * blockDim.x;

partialSum[t] = input[start + t];
partialSum[blockDim.x + t] = input[start + blockDim.x + t];

for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < 2 * stride) {
        partialSum[t] += partialSum[t + stride];
    }
}