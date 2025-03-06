
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  // Calculate the row index of the C element to work on
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Calculate the column index of the C element to work on
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure the thread is within bounds of C
  if (row < numCRows && col < numCColumns) {
    float sum = 0.0f;
    // Loop over the A row and B column to compute the dot product
    for (int k = 0; k < numAColumns; ++k) {
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    // Write the computed value to C
    C[row * numCColumns + col] = sum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA; // A matrix on device
  float *deviceB; // B matrix on device
  float *deviceC; // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;   // set to correct value
  numCColumns = numBColumns;   // set to correct value

  // Allocate memory for the output matrix C on the host
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here for A, B and C
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here for A and B
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // set block size to 16,16 and determine the grid dimensions
  // use dim3 structure for setting block and grid dimensions
  // Set block size to 16x16 and compute grid dimensions so that each thread computes one element in C
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((numCColumns + blockDim.x - 1) / blockDim.x,
               (numCRows + blockDim.y - 1) / blockDim.y,
               1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
    numARows, numAColumns,
    numBRows, numBColumns,
    numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
