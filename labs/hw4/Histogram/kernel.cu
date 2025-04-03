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
    const unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins)
{
    extern __shared__ unsigned int shared_bins[];


    // Compute global thread ID and total threads.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    
    // Coarsening parameter: number of elements processed per thread per batch.
    const unsigned int elements_per_thread = 8;  // Tunable parameter

    // Local accumulation in registers for compression.
    // We assume that in a batch a thread will encounter at most 'elements_per_thread' distinct bins.
    unsigned int local_bins[elements_per_thread];
    unsigned int local_counts[elements_per_thread];

    // Initialize local accumulators with an invalid marker.
    #pragma unroll
    for (unsigned int i = 0; i < elements_per_thread; i++) {
        local_bins[i] = num_bins;  // marker: invalid
        local_counts[i] = 0;
    }

    // Process input elements using a grid-stride loop over batches.
    for (unsigned int base = tid * elements_per_thread; base < num_elements; 
         base += total_threads * elements_per_thread) 
    {
        unsigned int num_local = 0;  // Number of distinct bins in this batch.
        // Process a batch of 'elements_per_thread' elements.
        #pragma unroll
        for (unsigned int j = 0; j < elements_per_thread; j++) {
            unsigned int idx = base + j;
            if (idx < num_elements) {
                unsigned int bin_idx = input[idx];
                if (bin_idx < num_bins) {
                    // Search for bin_idx in the local accumulator.
                    bool found = false;
                    for (unsigned int k = 0; k < num_local; k++) {
                        if (local_bins[k] == bin_idx) {
                            local_counts[k]++;
                            found = true;
                            break;
                        }
                    }
                    // If not found, add it if there is room.
                    if (!found) {
                        if (num_local < elements_per_thread) {
                            local_bins[num_local] = bin_idx;
                            local_counts[num_local] = 1;
                            num_local++;
                        } else {
                            // Flush the local accumulator to shared memory.
                            for (unsigned int k = 0; k < num_local; k++) {
                                atomicAdd(&shared_bins[local_bins[k]], local_counts[k]);
                                local_bins[k] = num_bins;
                                local_counts[k] = 0;
                            }
                            num_local = 0;
                            // Now add the current element.
                            local_bins[num_local] = bin_idx;
                            local_counts[num_local] = 1;
                            num_local++;
                        }
                    }
                }
            }
        }
        // Flush any remaining counts from the local accumulator to shared memory.
        for (unsigned int k = 0; k < num_local; k++) {
            atomicAdd(&shared_bins[local_bins[k]], local_counts[k]);
        }
    }
    __syncthreads();

    // Reduction Step: Accumulate shared histogram into global memory.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int count = shared_bins[i];
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