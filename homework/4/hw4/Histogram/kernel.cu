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

    // Initialize shared memory bins to zero
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }

    __syncthreads();

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
__global__ void histogram_shared_optimized(
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    // Padding to avoid shared memory bank conflicts (typically 32 banks)
    const unsigned int PADDING = 32;
    extern __shared__ unsigned int shared_bins[];

    // Initialize padded shared memory bins to zero cooperatively
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_bins[i + (i / PADDING)] = 0;
    }

    __syncthreads();

    // Calculate thread and grid dimensions for bucketing input
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Coarsening Step: explicitly handle multiple input elements per thread
    const unsigned int elements_per_thread = 4; // Tunable parameter
    unsigned int start = tid * elements_per_thread;
    unsigned int end = min(start + elements_per_thread, num_elements);

    // Populate local histogram in padded shared memory
    for (unsigned int i = start; i < end; ++i) {
        unsigned int bin_idx = input[i];
        if (bin_idx < num_bins) {
            atomicAdd(&(shared_bins[bin_idx + (bin_idx / PADDING)]), 1);
        }
    }

    __syncthreads();

    // Reduction Step: accumulate counts from padded shared memory into global memory
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int bin_count = shared_bins[i + (i / PADDING)];
        if (bin_count > 0) {
            atomicAdd(&(bins[i]), bin_count);
        }
    }

    // No further synchronization needed since each thread safely updates global bins independently
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