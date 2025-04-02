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


// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
// Version 2: Optimized Histogram Calculation using Shared Memory
// Detailed Comments Describing the Optimization Approach:
//
// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
// Version 2: Optimized Histogram Calculation using Shared Memory
// Detailed Comments Describing the Optimization Approach:
//
// Optimization strategy combines two key GPU parallel optimization techniques:
// 1. **Shared Memory Histogram**: Local histogram creation in shared memory to minimize global atomic conflicts.
// 2. **Coarsening**: Explicitly handling multiple input elements per thread to improve data locality and reduce overhead.
//
// The idea of thread coarsening is adapted from common GPU optimization techniques for parallel histogram computations described in CUDA programming best practices.

// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// Kernel 1: Each block computes its own partial histogram in shared memory.
__global__ void histogram_partial(
    const unsigned int *input,
    unsigned int *partial_hist, // each block writes its own histogram here
    unsigned int num_elements,
    unsigned int num_bins)
{
    // Allocate shared memory for the block’s histogram
    extern __shared__ unsigned int s_hist[];

    // Initialize shared memory histogram to zero cooperatively.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute global thread ID and overall stride.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Each thread processes a subset of the input.
    for (unsigned int i = tid; i < num_elements; i += stride) {
        unsigned int bin = input[i];
        if (bin < num_bins) {
            // Update shared memory histogram using atomic operations.
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // Write the block’s partial histogram to global memory.
    // Since each block writes to its own portion, no atomics are needed here.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        partial_hist[blockIdx.x * num_bins + i] = s_hist[i];
    }
}

// Kernel 2: Reduce the per-block histograms into the final histogram.
__global__ void histogram_reduce(
    const unsigned int *partial_hist, // array of partial histograms
    unsigned int *bins,               // final output histogram
    unsigned int num_blocks,
    unsigned int num_bins)
{
    // Each thread is responsible for summing one bin.
    unsigned int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < num_bins) {
        unsigned int sum = 0;
        // Sum this bin over all block-level histograms.
        for (unsigned int b = 0; b < num_blocks; b++) {
            sum += partial_hist[b * num_bins + bin];
        }
        bins[bin] = sum;
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