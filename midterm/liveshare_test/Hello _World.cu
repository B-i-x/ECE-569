//////Testing copilot


__global__ void helloWorld() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloWorld<<<1, 1>>>();
    
    // Wait for the GPU to finish before accessing the output
    cudaDeviceSynchronize();
    
    return 0;
}