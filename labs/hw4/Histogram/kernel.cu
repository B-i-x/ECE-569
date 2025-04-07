#include <limits>
#define UINT_MAX_VAL 0xFFFFFFFF
#define BITS_PER_PASS 8
#define RADIX (1 << BITS_PER_PASS)  // 256
#define MASK (RADIX - 1)            // 0xFF

// version 0
// Global memory only interleaved version.
// Compute global thread index and stride.
// Loop over input elements, then update the corresponding bin in global memory.
__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // global thread index
    int stride = blockDim.x * gridDim.x;                 // processing stride

    for (int j = idx; j < num_elements; j += stride) {
        int bin = input[j];                              // get bin index
        atomicAdd(&bins[bin], 1);                        // update global bin
    }
}

// version 1
// Shared memory privatized version.
// Compute global and block thread indices.
// Use shared memory for a private histogram and merge it into global bins.
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;  // global thread index
    int tid = threadIdx.x;                           // thread index within block

    extern __shared__ unsigned int s[];              // shared memory array

    // Clear shared memory bins. Loop covers cases where num_bins > blockDim.x.
    for (int j = tid; j < num_bins; j += blockDim.x) {
        s[j] = 0;
    }
    __syncthreads();  // wait for all threads to finish initialization

    // Process input elements and update private histogram in shared memory.
    for (int j = id; j < num_elements; j += blockDim.x * gridDim.x) {
        atomicAdd(&s[input[j]], 1);
    }
    __syncthreads();  // wait for all threads to finish updates

    // Merge shared histogram into global bins.
    for (int j = tid; j < num_bins; j += blockDim.x) {
        atomicAdd(&bins[j], s[j]);
    }
}

// version 2
// Shared memory optimized version with clipping.
// I compute my global thread ID and use shared memory to reduce global atomic updates.
__global__ void histogram_shared_optimized(
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    extern __shared__ unsigned int shared_bins[];

    // Clear shared memory bins.
    for (unsigned int j = threadIdx.x; j < num_bins; j += blockDim.x) {
        shared_bins[j] = 0;
    }
    __syncthreads();  // ensure all bins are cleared

    // Compute global thread ID and stride.
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Loop over input elements, check bounds, and update shared bins.
    for (unsigned int j = tid; j < num_elements; j += stride) {
        unsigned int bin_idx = input[j];
        if (bin_idx < num_bins) {
            atomicAdd(&shared_bins[bin_idx], 1);
        }
    }
    __syncthreads();  // wait for all threads to finish updating shared bins

    // Merge shared bins into global memory with clipping to 127.
    for (unsigned int j = threadIdx.x; j < num_bins; j += blockDim.x) {
        unsigned int bin_count = shared_bins[j];
        if (bin_count > 0) {
            unsigned int old = atomicAdd(&(bins[j]), bin_count);  // update global bin
            if (old + bin_count > 127) {                            // clip value to 127
                bins[j] = 127;
            }
        }
    }
}

// Clipping function: resets bins greater than 127 to 127.
// Loop over bins and clip any value above 127.
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int j = tid; j < num_bins; j += stride) {
        if (bins[j] > 127) {
            bins[j] = 127;
        }
    }
}
