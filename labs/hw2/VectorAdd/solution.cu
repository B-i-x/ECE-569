#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  // Compute the global index of the thread
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Ensure we do not access out-of-bounds elements
  if (idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  // Allocate GPU memory
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Copy input memory to the GPU
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Initialize the grid and block dimensions
  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;

  // Launch the GPU Kernel
  wbTime_start(Compute, "Performing CUDA computation");
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  // Copy the GPU memory back to the CPU
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Free the GPU memory
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
