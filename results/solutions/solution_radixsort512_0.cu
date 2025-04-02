// do not modify this file for histogram versions 0 and 1
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include "kernel.cu"
#define NUM_BINS 4096



#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void exclusiveScan(unsigned int *data, int length) {
  unsigned int sum = 0;
  for (int i = 0; i < length; i++) {
      unsigned int temp = data[i];
      data[i] = sum;
      sum += temp;
  }
}

void radix_sort(unsigned int **d_in, unsigned int **d_temp, int n) {
  // Allocate device memory for the histogram (256 ints) and bucket offsets.
  unsigned int *d_hist;
  CUDA_CHECK(cudaMalloc((void**)&d_hist, RADIX * sizeof(unsigned int)));
  
  unsigned int *d_bucket_offsets;
  CUDA_CHECK(cudaMalloc((void**)&d_bucket_offsets, RADIX * sizeof(unsigned int)));
  
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  
  // We'll perform passes for each 8 bits (i.e., 4 passes for 32 bits).
  int numPasses = (32 + BITS_PER_PASS - 1) / BITS_PER_PASS;
  // Temporary host buffer for histogram (256 ints).
  unsigned int h_hist[RADIX];
  
  // For each pass, sort by the current digit.
  for (int pass = 0; pass < numPasses; pass++) {
      int shift = pass * BITS_PER_PASS;
      
      // --- Step 1: Compute Histogram ---
      CUDA_CHECK(cudaMemset(d_hist, 0, RADIX * sizeof(unsigned int)));
      radix_histogram_kernel<<<gridSize, blockSize>>>(*d_in, n, shift, d_hist);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      
      // Copy histogram from device to host.
      CUDA_CHECK(cudaMemcpy(h_hist, d_hist, RADIX * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      
      // --- Step 2: Compute Exclusive Scan on Host ---
      exclusiveScan(h_hist, RADIX);
      
      // Copy the exclusive scan result (bucket offsets) to device.
      CUDA_CHECK(cudaMemcpy(d_bucket_offsets, h_hist, RADIX * sizeof(unsigned int), cudaMemcpyHostToDevice));
      
      // --- Step 3: Scatter Elements Based on the Current Digit ---
      radix_scatter_kernel<<<gridSize, blockSize>>>(*d_in, n, shift, d_bucket_offsets, *d_temp);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      
      // Swap the input and temporary pointers for the next pass.
      unsigned int *temp = *d_in;
      *d_in = *d_temp;
      *d_temp = temp;
  }
  
  // After the passes, if the number of passes is odd, the sorted array is in d_temp.
  // We want it in d_in, so copy back.
  if (numPasses & 1) {
      CUDA_CHECK(cudaMemcpy(*d_temp, *d_in, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      unsigned int *temp = *d_in;
      *d_in = *d_temp;
      *d_temp = temp;
  }
  
  // Free temporary device memory.
  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_bucket_offsets));
}




void histogram(
  unsigned int *input, 
  unsigned int *bins,
  unsigned int num_elements, 
  unsigned int num_bins, 
  int kernel_version) 
  {

 if (kernel_version == 0) {
  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_global_kernel<<<gridDim, blockDim>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
 }
 else if (kernel_version==1) {
 // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_shared_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
 }

 else if (kernel_version == 2) {
    // ----- Step 1: Sort the Input using Radix Sort -----
    // Here, 'input' is the unsorted device array and 'num_elements' is its length.
    // Allocate a temporary array of the same size.
    unsigned int *temp;
    CUDA_CHECK(cudaMalloc((void **)&temp, num_elements * sizeof(unsigned int)));
    
    // Call the custom radix sort.
    // Note: radix_sort takes pointers to the device pointers and will swap them as needed.
    radix_sort(&input, &temp, num_elements);
    // After this call, 'input' contains the sorted data.
    
    // Free the temporary array used by radix sort.
    CUDA_CHECK(cudaFree(temp));
    
    // ----- Step 2: Build the Histogram from the Sorted Input -----
    // Zero out bins before computing histogram.
    CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
    {
        // Launch the histogram_from_sorted kernel (previously implemented) to compute the histogram.
        dim3 blockDim(256);
        dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
        // We use 'num_elements' (original count) as the effective sorted length.
        histogram_from_sorted<<<gridDim, blockDim>>>(input, bins, num_elements, num_bins);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // ----- Step 3: Clip bin values if necessary -----
    {
        dim3 blockDim(512);
        dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
        convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
  }



}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int version; // kernel version global or shared 
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);
  
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput,
                        inputLength * sizeof(unsigned int)));
  CUDA_CHECK(
      cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput,
                        inputLength * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  // wbTime_start(Compute, "Performing CUDA computation");

  version = atoi(argv[5]); 
  cudaEventRecord(astartEvent, 0);
  histogram(deviceInput, deviceBins, inputLength, NUM_BINS,version);
  // wbTime_stop(Compute, "Performing CUDA computation");

  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f for version %d\n",aelapsedTime,version);
  printf("\n");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Verify correctness
  // -----------------------------------------------------
  printf ("Ran version %d\n", version);
  if (version == 0 )
     wbLog(TRACE, "Checking global memory only kernel");
  else if (version == 1) 
     wbLog(TRACE, "Checking shared memory kernel");
  else if (version == 2) 
     wbLog(TRACE, "Checking shared optimized kernel");
  wbSolution(args, hostBins, NUM_BINS);

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");


  free(hostBins);
  free(hostInput);
  return 0;
}
