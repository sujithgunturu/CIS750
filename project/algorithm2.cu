#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
void Algorithm2(int m, int n, int l);
// Block size used in algorithm 2 of GEMM
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 16
__device__ unsigned long long totThr = 0;

__global__ void device_Matrix_multi(const double* const device_matrix_A, const double* const device_matrix_B,double* device_matrix_C, const int m,const int n,const int l)
{
atomicAdd(&totThr, 1);
const int threadid_x = threadIdx.x + blockDim.x*blockIdx.x;
const int threadid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	
	if (threadid_x >= m || threadid_y >= l)
	{
		return;
	}
	int idx = threadid_y*m + threadid_x;
	double sum = 0.0;
	for (int k = 0; k < n; k++){
		int idxA = k*m + threadid_x;
		int idxB = threadid_y*n + k;	
		sum += device_matrix_A[idxA]*device_matrix_B[idxB];
			}
	device_matrix_C[idx] = sum;
}

int main()

{
Algorithm2(32, 32, 32);
Algorithm2(64, 64, 64);
Algorithm2(128,128, 128);
Algorithm2(256, 256, 256);
Algorithm2(512, 512, 512);
Algorithm2(1024, 1024, 1024);
Algorithm2(2048, 2048, 2048);
Algorithm2(4096, 4096, 4096);
	
}
void Algorithm2(const int m, const int n, const int l)
{
	double *device_matrix_A;
	double *device_matrix_B;
	double *device_matrix_C;
    double* matrix_A;
    double* matrix_B;
    double* matrix_C;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;



	
	// allocating the memory
	matrix_A = (double*)malloc( m*n* sizeof(double));
	matrix_B = (double*)malloc( m*l* sizeof(double));

	matrix_C = (double*)malloc( m*l*sizeof(double));

	cudaMalloc(&device_matrix_A, m*n*sizeof(double));
	cudaMalloc(&device_matrix_B, n*l*sizeof(double));
	cudaMalloc(&device_matrix_C, m*l*sizeof(double));
	for(int i = 0; i < m; i++)
	{	
		for(int j = 0; j <n; j++){
			matrix_A[i *n + j] = rand()%10;
			matrix_B[i *n + j] = rand()%10;
			matrix_C[i *n + j] = 0;
		}
	}

	
	// Copy data from the host memory to the device memory
	cudaMemcpy(device_matrix_A, matrix_A, m*n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_matrix_B, matrix_B, n*l*sizeof(double), cudaMemcpyHostToDevice);

	dim3 nthreads(0, 0);
	dim3 nblocks(0, 0);
	nthreads.x = BLOCK_SIZE_x;
	nthreads.y = BLOCK_SIZE_y;
	
	nblocks.x = (m + nthreads.x - 1)/nthreads.x;
	nblocks.y = (l + nthreads.y - 1)/nthreads.y;
	cudaEventRecord(start);
	// Launch the kernel
	device_Matrix_multi <<<nblocks, nthreads>>> (device_matrix_A,device_matrix_B,device_matrix_C,m,n,l);
	
	
	unsigned long long total;
  	cudaMemcpyFromSymbol(&total, totThr, sizeof(unsigned long long));
  	printf("Total threads counted in: %lu\n", total);	

	// Copy data from the device memory to the host memory
	cudaMemcpy(matrix_C, device_matrix_C, m*l*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	for(int i=0; i<m;i++){
		for(int j =0; j<n; j++){

					}
	}
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("elaspsed for algorithm 2 = %f ms\n\n\n", milliseconds);

	// Free the device memory
	cudaFree(device_matrix_A);
	cudaFree(device_matrix_B);
	
	cudaFree(device_matrix_C);
	free(matrix_A);
	free(matrix_B);

	free(matrix_C);

}


