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
  // ----- Step 1: Prepare sorted input -----
  // Compute padded length (next power-of-two)
  unsigned int padLength = 1;
  while (padLength < num_elements) {
      padLength <<= 1;
  }

  // Allocate device memory for the sorted input
  unsigned int *sortedInput;
  CUDA_CHECK(cudaMalloc((void **)&sortedInput, padLength * sizeof(unsigned int)));

  // Copy the original input into sortedInput
  CUDA_CHECK(cudaMemcpy(sortedInput, input, num_elements * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  // Pad the remaining elements with UINT_MAX so they sort to the end
  if (padLength > num_elements) {
      // Calculate number of bytes to pad
      size_t padBytes = (padLength - num_elements) * sizeof(unsigned int);
      // Create a host buffer filled with UINT_MAX_VAL
      unsigned int *hostPad = (unsigned int *)malloc(padBytes);
      for (unsigned int i = 0; i < (padLength - num_elements); i++) {
          hostPad[i] = UINT_MAX_VAL;
      }
      CUDA_CHECK(cudaMemcpy(sortedInput + num_elements, hostPad, padBytes, cudaMemcpyHostToDevice));
      free(hostPad);
  }

  // ----- Step 2: Sort the input using Bitonic Sort -----
  int block_size = 1024;
  int grid_size = (padLength + block_size - 1) / block_size;
  // Outer loop over subsequence sizes: k doubles each time.
  for (unsigned int k = 2; k <= padLength; k <<= 1) {
      // Inner loop: j starts at half of k and halves until 0.
      for (unsigned int j = k >> 1; j > 0; j >>= 1) {
          bitonic_sort_kernel<<<grid_size, block_size>>>(sortedInput, padLength, j, k);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
      }
  }

  // ----- Step 3: Compute Histogram from Sorted Input -----
  // Zero out bins before computing histogram.
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  {
      // Launch the histogram_from_sorted kernel.
      dim3 blockDim(1024);
      dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
      // Note: Use num_elements (original count) for binary search boundaries,
      // since padded values (UINT_MAX) are ignored for bins in [0, num_bins)
      histogram_from_sorted<<<gridDim, blockDim>>>(sortedInput, bins, num_elements, num_bins);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Free the temporary sorted input memory.
  CUDA_CHECK(cudaFree(sortedInput));

  // ----- Step 4: Clip bin values if necessary -----
  {
      dim3 blockDim(1024);
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
