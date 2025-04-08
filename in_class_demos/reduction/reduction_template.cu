/*
Template source code for experimenting with reduction functions covered in Modules 33-34 

load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o reduce reduction_template.cu

to execute: 
$ ./reduce > out.txt
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <chrono>   // C++11 chrono header for timing
#include <stdlib.h>

//-------------------------------------------------------------
// Kernel 0: Global memory reduction using stride-based approach.
// Uses the input array in global memory and performs in-place reduction.
// Note: This kernel is not optimal because of global memory accesses and divergent threads.
__global__ void global_reduce_stride(float * d_out, float * d_in)
{
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int n = blockDim.x;
    
    // Loop over strides: threads with indices that are multiples of (2*stride) add their partner.
    for (int stride = 1; stride < n; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0 && (tid + stride) < n)
        {
            d_in[blockStart + tid] += d_in[blockStart + tid + stride];
        }
        __syncthreads();
    }
    // Write the result for this block to output.
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[blockStart];
    }
}

//-------------------------------------------------------------
// Kernel 1: Shared memory reduction using stride-based approach.
// Each block loads its portion of global data into shared memory, then reduces it.
__global__ void shared_reduce_stride(float * d_out, float * d_in)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    
    // Load from global memory into shared memory.
    sdata[tid] = d_in[blockStart + tid];
    __syncthreads();
    
    // Reduction in shared memory using a stride loop.
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0 && (tid + stride) < blockDim.x)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block back to global memory.
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

//-------------------------------------------------------------
// Kernel 2: Shared memory reduction using a no-divergence approach.
// This kernel uses a loop that halves the number of active threads in each iteration.
__global__ void shared_reduce_stride_nodiverge(float * d_out, float * d_in)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    
    // Load input data into shared memory.
    sdata[tid] = d_in[blockStart + tid];
    __syncthreads();
    
    // Loop: each iteration halves the number of active threads.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // The first thread writes the block’s result.
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

//-------------------------------------------------------------
// Kernel 3: Shared memory reduction using a reversed tree approach.
// This kernel uses shared memory and performs the reduction by building the sum in the last element.
__global__ void shared_reduce_reverse(float * d_out, const float * d_in)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    
    // Load input element into shared memory.
    sdata[tid] = d_in[blockStart + tid];
    __syncthreads();
    
    int offset = 1;
    // Build the reduction tree in reverse (from the leftmost pair up to the last element).
    for (int d = blockDim.x >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    // Write the result (stored in the last element) to global memory.
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[blockDim.x - 1];
    }
}

//-------------------------------------------------------------
// Kernel 4: Shared reverse first reduction.
// This kernel loads and sums two elements per thread from global memory 
// (reducing the number of blocks by a factor of 2) and then performs the reversed tree reduction.
__global__ void shared_reverse_firstreduction(float * d_out, const float * d_in)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Each block processes two chunks of data. Calculate starting index.
    int blockStart = blockIdx.x * blockDim.x * 2;
    
    // Each thread loads two elements and adds them.
    sdata[tid] = d_in[blockStart + tid] + d_in[blockStart + tid + blockDim.x];
    __syncthreads();
    
    int offset = 1;
    // Reduction tree: sum in reversed order.
    for (int d = blockDim.x >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[blockDim.x - 1];
    }
}


//-------------------------------------------------------------
// Function that launches the reduction kernels.
// It assumes size is a multiple of maxThreadsPerBlock and no larger than maxThreadsPerBlock^2.
void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, int version)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    
    if (version==4)
    {
        global_reduce_stride<<<blocks, threads>>>(d_intermediate, d_in);
    }
    else if (version==3)
    {
        shared_reverse_firstreduction<<<blocks/2, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==2)
    {
        shared_reduce_reverse<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==1)
    {
        shared_reduce_stride_nodiverge<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==0)
    {
        shared_reduce_stride<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
     
    if (version==4)
    {
        global_reduce_stride<<<blocks, threads>>>(d_out, d_intermediate);
    }
    else if (version==3)
    {
        shared_reverse_firstreduction<<<blocks/2, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
    else if (version==2)
    {
        shared_reduce_reverse<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
    else if (version == 1)
    {
       shared_reduce_stride_nodiverge<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
        
    }
    else if (version == 0)
    {
        shared_reduce_stride<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
}
int main(int argc, char **argv)
{
    int j;
    float nsum;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // Generate the input array on the host.
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // Generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)rand() / ((float)RAND_MAX / 2.0f);
        sum += h_in[i];
    }

    // Declare GPU memory pointers.
    float * d_in, * d_intermediate, * d_out;

    // Allocate GPU memory.
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(float));

    // Transfer the input array to the GPU.
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;      
    cudaEvent_t start, stop;

    // Use chrono to measure the serial (CPU) sum time.
    nsum = 0.0f;
    auto host_start = std::chrono::high_resolution_clock::now();
    for (j = 0; j < ARRAY_SIZE; j++) {
        nsum += h_in[j];
    }
    auto host_end = std::chrono::high_resolution_clock::now();
    double host_elapsed_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();

    printf("serial code execution time %f ms\n", host_elapsed_ms);
    printf("--------------------------\n\n");

    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
         printf("Using device %d:\n", dev);
         printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
                devProps.name, (int)devProps.totalGlobalMem, 
                (int)devProps.major, (int)devProps.minor, 
                (int)devProps.clockRate);
    }
  
    // Run the different reduction kernels.
    for (whichKernel = 0; whichKernel < 5; whichKernel++) 
    { 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // Launch the kernel for the chosen version.
        switch (whichKernel) {
        case 0:
            printf("Running shared stride reduce\n");
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++)
            {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 0);
            }
            cudaEventRecord(stop, 0);
            break;
        case 1:
            printf("Running shared stride no divergent reduce\n");
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++)
            {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 1);
            }
            cudaEventRecord(stop, 0);
            break;
        case 2:
            printf("Running shared reduce reversed\n");
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++)
            {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 2);
            }
            cudaEventRecord(stop, 0);
            break;
        case 3:
            printf("Running global reduce stride - naive first reduction\n");
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++)
            {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 3);
            }
            cudaEventRecord(stop, 0);
            break;
        case 4:
            printf("Running global reduce stride - naive\n");
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++)
            {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 4);
            }
            cudaEventRecord(stop, 0);
            break;
        default:
            fprintf(stderr, "error: ran no kernel\n");
            exit(EXIT_FAILURE);
        }
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);    
        elapsedTime /= 100.0f;      // average over 100 trials

        // Copy back the sum from GPU.
        float h_out;
        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("average time elapsed >>>>>>> %f ms\n", elapsedTime);
        printf("\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // Free GPU memory allocation.
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
        
    return 0;
}