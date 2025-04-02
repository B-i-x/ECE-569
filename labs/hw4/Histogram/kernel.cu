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
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) 
    {
    extern __shared__ unsigned int shared_bins[];

    // Initialize shared memory histogram to zero
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    // Process a fixed number of elements per thread iteration
    const unsigned int elements_per_thread = 8;  // Tunable parameter

    // Local registers for compression
    // Assuming worst-case: each element in the batch falls into a different bin.
    unsigned int local_bins[elements_per_thread];
    unsigned int local_counts[elements_per_thread];

    // Initialize local accumulators
    for (int i = 0; i < elements_per_thread; i++) {
        local_bins[i] = num_bins;  // Use an invalid bin index as placeholder
        local_counts[i] = 0;
    }

    // Grid-stride loop to cover all input elements in batches
    for (unsigned int base = tid * elements_per_thread; base < num_elements; base += total_threads * elements_per_thread) {
        // Reset local accumulators for this batch
        int num_local = 0;
        for (int i = 0; i < elements_per_thread; i++) {
            unsigned int index = base + i;
            if (index >= num_elements) break;

            unsigned int bin = input[index];
            // Check if bin is already in our local accumulator
            bool found = false;
            for (int j = 0; j < num_local; j++) {
                if (local_bins[j] == bin) {
                    local_counts[j]++;
                    found = true;
                    break;
                }
            }
            // If not found, add a new entry if there is room
            if (!found && num_local < elements_per_thread) {
                local_bins[num_local] = bin;
                local_counts[num_local] = 1;
                num_local++;
            }
        }
        // Perform fewer atomic updates: one per unique bin in this batch
        for (int i = 0; i < num_local; i++) {
            if (local_bins[i] < num_bins) {
                atomicAdd(&shared_bins[local_bins[i]], local_counts[i]);
            }
        }
    }
    __syncthreads();

    // Reduction: accumulate shared histogram into global histogram
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
            atomicAdd(&bins[i], shared_bins[i]);
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