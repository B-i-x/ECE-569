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
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) {

// Declare shared memory histogram.
extern __shared__ unsigned int shared_bins[];

// Initialize shared memory histogram bins to 0.
for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    shared_bins[i] = 0;
}
__syncthreads();

// Determine global thread index and input range.
unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int elements_per_thread = 4;  // Tunable parameter
unsigned int start = tid * elements_per_thread;
unsigned int end = min(start + elements_per_thread, num_elements);

// Define a small register accumulation structure.
// We use REG_SIZE slots. A slot holds a bin index and the count accumulated in that register.
const int REG_SIZE = 4; // Tunable; must be small so it fits in registers.
int reg_bins[REG_SIZE];
unsigned int reg_counts[REG_SIZE];

// Initialize register accumulation slots to a sentinel value (here -1).
#pragma unroll
for (int j = 0; j < REG_SIZE; j++) {
    reg_bins[j] = -1;
    reg_counts[j] = 0;
}

// Process the input elements assigned to this thread.
for (unsigned int i = start; i < end; i++) {
    unsigned int bin_idx = input[i];
    if (bin_idx >= num_bins) continue;

    bool found = false;
    // Check if the current bin is already in the register array.
    #pragma unroll
    for (int j = 0; j < REG_SIZE; j++) {
        if (reg_bins[j] == bin_idx) {
            reg_counts[j]++;
            found = true;
            break;
        }
    }
    // If not found, try to find an empty slot.
    if (!found) {
        bool inserted = false;
        #pragma unroll
        for (int j = 0; j < REG_SIZE; j++) {
            if (reg_bins[j] == -1) {
                reg_bins[j] = bin_idx;
                reg_counts[j] = 1;
                inserted = true;
                break;
            }
        }
        // If no empty slot is available, flush the register accumulation to shared memory.
        if (!inserted) {
            #pragma unroll
            for (int j = 0; j < REG_SIZE; j++) {
                if (reg_bins[j] != -1) {
                    atomicAdd(&(shared_bins[reg_bins[j]]), reg_counts[j]);
                    // Reset the register slot.
                    reg_bins[j] = -1;
                    reg_counts[j] = 0;
                }
            }
            // Insert the current bin into the first slot.
            reg_bins[0] = bin_idx;
            reg_counts[0] = 1;
        }
    }
}

// Flush any remaining register accumulators to shared memory.
#pragma unroll
for (int j = 0; j < REG_SIZE; j++) {
    if (reg_bins[j] != -1) {
        atomicAdd(&(shared_bins[reg_bins[j]]), reg_counts[j]);
    }
}
__syncthreads();

// Reduction Step: Each thread reduces a portion of the shared memory histogram into global memory.
for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    unsigned int bin_count = shared_bins[i];
    if (bin_count > 0) {
        atomicAdd(&(bins[i]), bin_count);
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