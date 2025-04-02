#include <limits>
#define UINT_MAX_VAL 0xFFFFFFFF
#define BITS_PER_PASS 8
#define RADIX (1 << BITS_PER_PASS)  // 256
#define MASK (RADIX - 1)            // 0xFF
// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(
    unsigned int *input, unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    // Calculate the global thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Iterate over elements in an interleaved pattern
    for (unsigned int i = tid; i < num_elements; i += stride) {
        unsigned int bin_idx = input[i];

        // Ensure the bin index is within bounds
        if (bin_idx < num_bins) {
            // Update bin count atomically to prevent race conditions
            atomicAdd(&(bins[bin_idx]), 1);
        }
    }
}


// Version 1: Histogram calculation using shared memory privatization
// Approach:
// 1. Allocate a shared memory array (private histogram) per block.
// 2. Each thread increments counts within shared memory using atomic operations to avoid race conditions.
// 3. After computation, shared memory histograms are combined into global memory.

__global__ void histogram_shared_kernel(
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    extern __shared__ unsigned int shared_bins[];


    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Populate the shared histogram
    for (unsigned int i = tid; i < num_elements; i += stride) {
        unsigned int bin_idx = input[i];
        if (bin_idx < num_bins) {
            atomicAdd(&(shared_bins[bin_idx]), 1);
        }
    }

    __syncthreads();

    // Write shared histogram to global histogram
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&(bins[i]), shared_bins[i]);
    }
}



__global__ void histogram_shared_optimized(
    const unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins)
{
    extern __shared__ unsigned int shared_bins[];

    // Compute global thread index and total number of threads.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    // Coarsening: each thread processes multiple elements.
    // Tunable parameter: process 8 elements per thread.
    const unsigned int elements_per_thread = 4; 
    for (unsigned int base = tid * elements_per_thread; base < num_elements; base += total_threads * elements_per_thread) {
        // Unroll the loop over the batch.
        #pragma unroll
        for (unsigned int j = 0; j < elements_per_thread; j++) {
            unsigned int idx = base + j;
            if (idx < num_elements) {
                unsigned int bin_idx = input[idx];
                if (bin_idx < num_bins) {
                    atomicAdd(&shared_bins[bin_idx], 1);
                }
            }
        }
    }
    __syncthreads();

    // Reduction Step: each thread writes its portion of the shared histogram to global memory.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int count = shared_bins[i];
        // Only perform atomicAdd if there's something to add.
        if (count > 0) {
            atomicAdd(&bins[i], count);
        }
    }
}



// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127

// Clipping function: resets bins greater than 127 to 127
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < num_bins; i += stride) {
        if (bins[i] > 127) {
            bins[i] = 127;
        }
    }
}