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
__global__ void histogram_shared_optimized(
    unsigned int *input, 
    unsigned int *bins,
    unsigned int num_elements,
    unsigned int num_bins) 
    {
    extern __shared__ unsigned int shared_bins[];

    // Calculate thread and grid dimensions for bucketing input
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Coarsening Step: explicitly handle multiple input elements per thread
    const unsigned int elements_per_thread = 4; // Tunable parameter
    unsigned int start = tid * elements_per_thread;
    unsigned int end = min(start + elements_per_thread, num_elements);

    // Populate local histogram in shared memory
    for (unsigned int i = start; i < end; ++i) {
        unsigned int bin_idx = input[i];
        if (bin_idx < num_bins) {
            atomicAdd(&(shared_bins[bin_idx]), 1);
        }
    }

    __syncthreads();

    // Reduction Step: accumulate counts from shared memory into global memory
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int bin_count = shared_bins[i];
        if (bin_count > 0) {
            atomicAdd(&(bins[i]), bin_count);
        }
    }

    // No further synchronization needed since each thread safely updates global bins independently
}


// -----------------------------------------------------------------
// Bitonic Sort Kernel
// This kernel performs one comparison-swap step for the bitonic sort.
// It uses two parameters, 'j' and 'k', which control the current stage.
// The array 'd_data' is assumed to have 'length' elements (a power of two).
__global__ void bitonic_sort_kernel(unsigned int *d_data, int length, int j, int k) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    // Compute the partner index for this thread.
    unsigned int ixj = idx ^ j;
    if (ixj > idx && ixj < length) {
        // For indices with bit (k) not set, sort in ascending order.
        if ((idx & k) == 0) {
            if (d_data[idx] > d_data[ixj]) {
                unsigned int temp = d_data[idx];
                d_data[idx] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
        // Otherwise, sort in descending order.
        else {
            if (d_data[idx] < d_data[ixj]) {
                unsigned int temp = d_data[idx];
                d_data[idx] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
    }
}

// -----------------------------------------------------------------
// Device functions: Binary search helpers for lower_bound and upper_bound.
// These are used by the histogram_from_sorted kernel.
__device__ unsigned int lower_bound(const unsigned int *data, unsigned int n, unsigned int key) {
    unsigned int low = 0;
    unsigned int high = n;
    while (low < high) {
        unsigned int mid = low + (high - low) / 2;
        if (data[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__device__ unsigned int upper_bound(const unsigned int *data, unsigned int n, unsigned int key) {
    unsigned int low = 0;
    unsigned int high = n;
    while (low < high) {
        unsigned int mid = low + (high - low) / 2;
        if (data[mid] <= key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}
// Kernel 1: Compute the histogram for the current digit.
// For each element in 'in', extract the digit at position 'shift'
// and atomically increment the corresponding bucket in 'hist'.
__global__ void radix_histogram_kernel(const unsigned int *in, int n, int shift, unsigned int *hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        unsigned int value = in[idx];
        unsigned int digit = (value >> shift) & MASK;
        atomicAdd(&hist[digit], 1);
    }
}

// -----------------------------------------------------------------
// Kernel 2: Scatter elements into the output array using the bucket offsets.
// Each element's digit is computed and then an atomicAdd on bucket_offsets
// yields the proper position at which to write the element into 'out'.
__global__ void radix_scatter_kernel(const unsigned int *in, int n, int shift, unsigned int *bucket_offsets, unsigned int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        unsigned int value = in[idx];
        unsigned int digit = (value >> shift) & MASK;
        // Atomically update the bucket offset for this digit and get the position.
        unsigned int pos = atomicAdd(&bucket_offsets[digit], 1);
        out[pos] = value;
    }
}

// -----------------------------------------------------------------
// Histogram from Sorted Kernel
// Each thread is responsible for one bin. It uses binary search (lower_bound and upper_bound)
// on the sorted input to determine how many elements fall into its bin.
__global__ void histogram_from_sorted(const unsigned int *sorted_input,
                                        unsigned int *bins,
                                        unsigned int num_elements,
                                        unsigned int num_bins) {
    unsigned int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < num_bins) {
         unsigned int lb = lower_bound(sorted_input, num_elements, bin);
         unsigned int ub = upper_bound(sorted_input, num_elements, bin);
         bins[bin] = ub - lb;
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