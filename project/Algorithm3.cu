#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


#include <cuda.h>
//void Algorithm1();
void Algorithm3(int m, int n, int l);
//for gemm 3 algorithm
#define BLOCK_SIZE 32
__device__ unsigned long long totThr = 0;
template <int block_size>
__global__ void device_Matrix_multi(
				    const double* const device_matrix_A,const double* const device_matrix_B,double* device_matrix_C,const int m,const int n,const int l)
{

atomicAdd(&totThr, 1);
const int global_threadid_x = threadIdx.x + blockDim.x*blockIdx.x;
const int global_threadid_y = threadIdx.y + blockDim.y*blockIdx.y;

const int threadid_x = threadIdx.x;
const int threadid_y = threadIdx.y;

const int blockid_x = blockIdx.x;
const int blockid_y = blockIdx.y;

__shared__ double matrix_A_shared[block_size+1][block_size+1];
__shared__ double matrix_B_shared[block_size+1][block_size+1];

int matrix_A_start = block_size*blockid_x;
int matrix_A_iteration  = block_size*m;
int matrix_B_start = n*block_size*blockid_y;
int matrix_B_end   = matrix_B_start + n - 1;
int matrix_B_iteration  = block_size;
double sum = 0.0;
int idxAcol = 0;
int idxBrow = 0; 			
for (int idxA = matrix_A_start, idxB = matrix_B_start;idxB <= matrix_B_end;idxA += matrix_A_iteration, idxB += matrix_B_iteration){
	if (global_threadid_x < m && threadid_y + idxAcol < n)
		matrix_A_shared[threadid_x][threadid_y] = device_matrix_A[idxA + m*threadid_y + threadid_x];
	if (threadid_x + idxBrow < n && global_threadid_y < l)
		matrix_B_shared[threadid_x][threadid_y] = device_matrix_B[idxB + n*threadid_y + threadid_x];
__syncthreads();				
	

	
	if (global_threadid_x < m && global_threadid_y < l){
		int boundary = min(block_size, n - idxAcol);		
		for (int k = 0; k < boundary; k++){
			sum += matrix_A_shared[threadid_x][k]*matrix_B_shared[k][threadid_y];
			}
	}
				
	idxAcol += block_size;
	idxBrow += block_size;
	__syncthreads();
	}
			
	if (global_threadid_x < m && global_threadid_y < l){
		int idxD = m*block_size*blockid_y + block_size*blockid_x + threadid_x + m*threadid_y;
		device_matrix_C[idxD] = sum;	
	}
}

int main()

{
Algorithm3(32, 32, 32);
Algorithm3(64, 64, 64);
Algorithm3(128,128, 128);
Algorithm3(256, 256, 256);
Algorithm3(512, 512, 512);
Algorithm3(1024, 1024, 1024);
Algorithm3(2048, 2048, 2048);
Algorithm3(4096, 4096, 4096);

	
}
void Algorithm3(int m, int n, int l)	{
	printf("inside function 3");
	
	double*  matrix_A;
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
	nthreads.x = BLOCK_SIZE;
	nthreads.y = BLOCK_SIZE;

	nblocks.x = (m + nthreads.x - 1)/nthreads.x;
	nblocks.y = (l + nthreads.y - 1)/nthreads.y;

	// Launch the kernel
	cudaEventRecord(start);
	device_Matrix_multi <BLOCK_SIZE><<<nblocks, nthreads>>> (  
							
							device_matrix_A,
							device_matrix_B,
							
							device_matrix_C,
							m,
							n,
							l);
	unsigned long long total;
  	cudaMemcpyFromSymbol(&total, totThr, sizeof(unsigned long long));
  	printf("Total threads counted in mat 3: %lu\n", total);	


	
	// Copy data from the device memory to the host memory
	cudaMemcpy(matrix_C, device_matrix_C, m*l*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);  
	for(int i=0; i<m;i++){
		for(int j =0; j<n; j++){
			
		}
	}
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("elaspsed = %f ms\n\n\n", milliseconds);

	// Free the device memory
	cudaFree(device_matrix_A);
	cudaFree(device_matrix_B);
	cudaFree(device_matrix_C);
	
	free(matrix_A);
	free(matrix_B);
	free(matrix_C);
	
	}


