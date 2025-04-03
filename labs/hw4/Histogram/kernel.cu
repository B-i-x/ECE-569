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

    extern __shared__  unsigned int sdata[];

    // Compute a 16-byte aligned pointer from sdata.
    unsigned int *shared_bins = (unsigned int*)(((uintptr_t)sdata + 15) & ~(uintptr_t)15);

    // Use a vectorized int4 pointer.
    int4 *shared_bins_int = (int4 *)shared_bins;
    unsigned int vec_length = num_bins / 4;  // each int4 covers 4 bins

    // Initialize the shared memory to zero.
    for (unsigned int i = threadIdx.x; i < vec_length; i += blockDim.x) {
        shared_bins_int[i] = make_int4(0, 0, 0, 0);
    }
    __syncthreads();

    // Calculate global thread id and total number of threads.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    const unsigned int elements_per_thread = 4; // Process 4 elements per iteration

    // Coarsening Step: each thread processes multiple groups of elements.
    for (unsigned int i = tid * elements_per_thread; i < num_elements; i += total_threads * elements_per_thread) {
        unsigned int end = min(i + elements_per_thread, num_elements);
        for (unsigned int j = i; j < end; j++) {
            unsigned int bin_idx = input[j];
            if (bin_idx < num_bins) {
                atomicAdd(&(shared_bins[bin_idx]), 1);
            }
        }
    }
    __syncthreads();

    // Reduction Step: accumulate counts from shared memory into global memory with loop unrolling.
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x * 4) {
        unsigned int count0 = shared_bins[i];
        unsigned int count1 = (i + blockDim.x < num_bins) ? shared_bins[i + blockDim.x] : 0;
        unsigned int count2 = (i + 2 * blockDim.x < num_bins) ? shared_bins[i + 2 * blockDim.x] : 0;
        unsigned int count3 = (i + 3 * blockDim.x < num_bins) ? shared_bins[i + 3 * blockDim.x] : 0;
        
        if (count0 > 0)
            atomicAdd(&bins[i], count0);
        if (count1 > 0)
            atomicAdd(&bins[i + blockDim.x], count1);
        if (count2 > 0)
            atomicAdd(&bins[i + 2 * blockDim.x], count2);
        if (count3 > 0)
            atomicAdd(&bins[i + 3 * blockDim.x], count3);
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