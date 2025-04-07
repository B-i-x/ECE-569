//Version-1) kernel_1t1e: each thread produces one output matrix using vertical and horizontal position as index

__global__ void kernel_1t1e(
    float *A, float *B, float *C, unsigned long WIDTH) 
    {
        int rowID = threadIdx.y + blockIdx.y * blockDim.y;
        int colID = threadIdx.x + blockIdx.x * blockDim.x;
        int elemID;
        if(rowID < WIDTH && colID < WIDTH){                           
            elemID = colID + rowID * WIDTH;
            C[elemID] = A[elemID] + B[elemID];
            }
    }
//Version-2) kernel_1t1r: each thread produces 1 output row using horizontal position as index
__global__ void kernel_1t1r(
    float *A, float *B, float *C, unsigned long WIDTH)
    {              
        int rowID = threadIdx.x + blockIdx.x * blockDim.x;

        if(rowID < WIDTH) {                           
            for(int i = 0; i<WIDTH;i++) {
                C[i + rowID*WIDTH] = A[i + rowID*WIDTH] + B[i + rowID*WIDTH];
            }
        }
    }

//Version-3) kernel_1t1c: each thread produces 1 output column using horizontal position as index
__global__ void kernel_1t1c(
    float *A, float *B, float *C, unsigned long WIDTH) 
    {              
        int colID = threadIdx.x + blockIdx.x * blockDim.x; // Row address
            if(colID < WIDTH) {
                for(int i = 0; i<WIDTH; i++){
                    C[colID + i*WIDTH] = A[colID + i*WIDTH] + B[colID + i*WIDTH];                         
                }
            }
    }