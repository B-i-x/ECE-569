#include <limits>
#define UINT_MAX_VAL 0xFFFFFFFF
#define BITS_PER_PASS 8
#define RADIX (1 << BITS_PER_PASS)  // 256
#define MASK (RADIX - 1)            // 0xFF
// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

// insert your code here
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < num_elements; i += stride) {
int bin = input[i];
atomicAdd(&bins[bin], 1);
}
}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {
    // Compute the global thread index and the thread index within the block.
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Declare shared memory dynamically.
    extern __shared__ unsigned int s[];

    // Each thread initializes one or more shared memory locations (bins) to zero.
    // Loop in case num_bins > blockDim.x.
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s[i] = 0;
    }
    // Ensure that shared memory initialization is complete before any thread uses it.
    __syncthreads();

    // Each thread processes one element from the input array if within bounds.
    // The input value is used as an index to increment the corresponding bin in shared memory.
    // Atomic operation ensures correctness if multiple threads write to the same bin.
    for (int i = id; i < num_elements; i += blockDim.x * gridDim.x) {
        atomicAdd(&s[input[i]], 1); 
    }
    // Wait for all threads to complete updating the shared histogram.
    __syncthreads();

    // Each thread adds its portion of the shared histogram into the global bins array.
    // Loop in case num_bins > blockDim.x.
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], s[i]);
    }
}

__global__ void histogram_shared_optimized(
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    extern __shared__ unsigned int shared_bins[];

    // Initialize shared memory bins to zero cooperatively
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }

    __syncthreads();

    // Calculate thread and grid dimensions for bucketing input
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Coarsening Step: explicitly handle multiple input elements per thread
    const unsigned int elements_per_thread = 4; // Tunable parameter

    // Use a grid-stride loop to cover all input elements:
    for (unsigned int base = tid * elements_per_thread; base < num_elements; base += total_threads * elements_per_thread) {
        // Calculate the end index for this batch.
        unsigned int end = base + elements_per_thread;
        if (end > num_elements) end = num_elements;
        // Process this batch.
        for (unsigned int i = base; i < end; ++i) {
            unsigned int bin_idx = input[i];
            if (bin_idx < num_bins) {
                atomicAdd(&(shared_bins[bin_idx]), 1);
            }
        }
    }
    __syncthreads();

    // Reduction Step: each thread reduces a portion of the shared histogram into global memory.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int bin_count = shared_bins[i];
        if (bin_count > 0) {
            atomicAdd(&(bins[i]), bin_count);
        }
    }
    // No further synchronization is needed.
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