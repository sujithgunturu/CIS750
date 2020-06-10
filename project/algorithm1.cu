#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>


void Algorithm1(int m, int n, int l);
#define BLOCK_SIZE 256


__global__ void device_Matrix_multi(const double* const device_matrix_A,const double* const device_matrix_B,
				    
				    double* device_matrix_C,
				    const int m,
				    const int n,
				    const int l
				    )
{


	int threadid = threadIdx.x + blockDim.x*blockIdx.x;
	int column = threadid%l;
	int row = threadid/l;
	if (row >= m || column >= l)
	{
		return;
	}
	int idx = column*m + row;
	double sum = 0.0;
	for (int k = 0; k < n; k++){
		int idxA = k*m + row;
		int idxB = column*n + k;	
		sum += device_matrix_A[idxA]*device_matrix_B[idxB];
	}
	device_matrix_C[idx] = sum;
			
}

int main()

{
Algorithm1(32, 32, 32);
Algorithm1(64, 64, 64);
Algorithm1(128,128, 128);
Algorithm1(256, 256, 256);
Algorithm1(512, 512, 512);
Algorithm1(1024, 1024, 1024);
Algorithm1(2048, 2048, 2048);
Algorithm1(4096, 4096, 4096);	
}
void Algorithm1(int m, int n, int l)	{
	printf("inside function");
	double* matrix_A;
    double* matrix_B;

    double* matrix_C;
	double *device_matrix_A;
	double *device_matrix_B;

	double *device_matrix_C;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	// Allocate the device memory
	matrix_A = (double*)malloc( m*n* sizeof(double));
	matrix_B = (double*)malloc( m*l* sizeof(double));
	matrix_C = (double*)malloc( m*l*sizeof(double));

	cudaMalloc(&device_matrix_A, m*n*sizeof(double));
	cudaMalloc(&device_matrix_B, n*l*sizeof(double));
	cudaMalloc(&device_matrix_C, m*l*sizeof(double));
	//initializing the matrixrix// Omit D and C
	for(int i = 0; i < m; i++)
	{	
		for(int j = 0; j <n; j++){
			matrix_A[i *n + j] = rand()%10;
			matrix_B[i *n + j] = rand()%10;
			matrix_C[i *n + j] = 0;
		
		}
	}
	cudaMemcpy(device_matrix_A, matrix_A, m*n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_matrix_B, matrix_B, n*l*sizeof(double), cudaMemcpyHostToDevice);

	int num_blocks = (m*l + BLOCK_SIZE - 1)/BLOCK_SIZE;
	cudaEventRecord(start);
	device_Matrix_multi <<<num_blocks, BLOCK_SIZE>>> (  
							device_matrix_A,
							device_matrix_B,
							device_matrix_C,
							m,
							n,
							l
							);
	cudaMemcpy(matrix_C, device_matrix_C, m*l*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);  
	for(int i=0; i<m;i++){
		for(int j =0; j<n; j++){
			//print to check whether the matrixrixs give correct output or not  
		}
	}
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("elaspsed = %f ms\n",milliseconds);
	cudaFree(device_matrix_A);
	cudaFree(device_matrix_B);

	cudaFree(device_matrix_C);
	free(matrix_A);
	free(matrix_B);
	
	free(matrix_C);
	
	}


