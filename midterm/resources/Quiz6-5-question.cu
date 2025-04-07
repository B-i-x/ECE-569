unsigned int t = threadIdx.x;
unsigned int start = 2 * blockIdx.x * blockDim.x;

partialSum[t] = input[start + t];
partialSum[blockDim.x + t] = input[start + blockDim.x + t];

for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
 

    atomicAdd(Total, 1);
    atomicAdd(&Total, &Partial);
    atomicAdd(Total, &Partial);
    atomicAdd(&Total, Partial);


    __syncthreads();

    if (t % (2 * stride) == 0) {
        partialSum[2 * t] += partialSum[2 * t + stride];
    }
}