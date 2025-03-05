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

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  // Define shared memory tiles for A and B.
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // Identify the block and thread indices.
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Calculate the row and column index of the C element to work on.
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  // Accumulate the dot product in a register.
  float sum = 0.0f;

  // Loop over all the tiles needed to cover the A matrix's columns / B matrix's rows.
  int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int t = 0; t < numTiles; t++) {
    // Load the tile from A into shared memory.
    int tiledACol = t * TILE_WIDTH + tx;
    if (row < numARows && tiledACol < numAColumns)
      As[ty][tx] = A[row * numAColumns + tiledACol];
    else
      As[ty][tx] = 0.0f;  // Pad with zeros if out of bounds

    // Load the tile from B into shared memory.
    int tiledBRow = t * TILE_WIDTH + ty;
    if (tiledBRow < numBRows && col < numBColumns)
      Bs[ty][tx] = B[tiledBRow * numBColumns + col];
    else
      Bs[ty][tx] = 0.0f;  // Pad with zeros if out of bounds

    // Synchronize to ensure all threads have loaded their data.
    __syncthreads();

    // Multiply the two tiles together and accumulate the results.
    for (int k = 0; k < TILE_WIDTH; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to ensure that computation is done before loading new tiles.
    __syncthreads();
  }

  // Write the computed value to C if within matrix bounds.
  if (row < numCRows && col < numCColumns) {
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
  int numCRows;    // number of rows in the matrix C(you have to set this)
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
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");


  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13. 
  // Copy A and B from host to device.
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");


  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numCRows + TILE_WIDTH - 1) / TILE_WIDTH,
               1);


  //@@ Launch the GPU Kernel here
  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
    numARows, numAColumns,
    numBRows, numBColumns,
    numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");


  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
  cudaMemcpyDeviceToHost);
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
