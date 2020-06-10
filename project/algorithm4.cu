#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


#include <cuda.h>
//void Algorithm1();
void Algorithm4(int m, int n, int l);

//for gemm 4 algorithm
#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 4

template<int block_size_x, int block_size_y>
__global__ void device_Matrix_multi(const double* const device_matrix_A,const double* const device_matrix_B,double* device_matrix_C,const int m,const int n,const int l)
{


	const int threadid_x = threadIdx.x;
	const int threadid_y = threadIdx.y;
	const int blockid_x = blockIdx.x;
	const int blockid_y = blockIdx.y;
	__shared__ double matrix_B_shared[block_size_x][block_size_x+1];
	double c[block_size_x];
	for (int i = 0; i< block_size_x; i++)
	{
		c[i] = 0.0;
	}
		int idx_A = blockid_x*block_size_x*block_size_y + threadid_x + threadid_y*block_size_x;
		int idx_B = threadid_x + (blockid_y*block_size_x + threadid_y)*n;
		int idx_B_last = idx_B + n;
		int col_A = 0;
		
		do
		{

			for(int i = 0; i < block_size_x; i += block_size_y)
				matrix_B_shared[threadid_x][threadid_y + i] = device_matrix_B[idx_B + i*n];
			
			idx_B += block_size_x;
			
			__syncthreads();
			
			int i_bound = min(block_size_x, n - col_A);
			for (int i = 0; i < i_bound; i++, idx_A+=m)
			{

				for (int j = 0; j < block_size_x; j++)
				{
					c[j] += device_matrix_A[idx_A]*matrix_B_shared[i][j];
				}
			}
			
			col_A += block_size_x;
			
			__syncthreads();
		
		}while (idx_B < idx_B_last);
	
	
	if (blockid_x*block_size_x*block_size_y + threadid_x + threadid_y*block_size_x < m)
	{
		int idx_D = blockid_x*block_size_x*block_size_y + (threadid_x + threadid_y*block_size_x) + blockid_y*block_size_x*m;
	
			int i_bound = min(block_size_x, l - blockid_y*block_size_x);
			for (int i = 0; i < i_bound; i++, idx_D += m)
			{
				device_matrix_C[idx_D] = c[i];
			}
		
		
	
	}
}

int main()

{
Algorithm4(32, 32, 32);
Algorithm4(64, 64, 64);
Algorithm4(128,128, 128);
Algorithm4(256, 256, 256);
Algorithm4(512, 512, 512);
Algorithm4(1024, 1024, 1024);
Algorithm4(2048, 2048, 2048);
Algorithm4(4096, 4096, 4096);

	
}
void Algorithm4(int m, int n, int l)	{
	printf("inside function");
	
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
	dim3 nblocks(0, 0);	// Launch the kernel

	nthreads.x = BLOCK_SIZE_x;
	nthreads.y = BLOCK_SIZE_y;

	nblocks.x = (m + nthreads.x*nthreads.y - 1)/(nthreads.x*nthreads.y);
	nblocks.y = (l + nthreads.x - 1)/nthreads.x;

	cudaEventRecord(start);
	printf("nuumber of blocks in x = %d\n", nblocks.x);
	printf("nuumber of blocks in y = %d\n", nblocks.y);
	printf("number of threads in x = %d\n", nthreads.x);
	printf("number of threads in y =%d\n", nthreads.y);
	printf("total threads = %d", nblocks.x*nblocks.y*nthreads.x*nthreads.y);
	device_Matrix_multi<BLOCK_SIZE_x, BLOCK_SIZE_y> <<<nblocks, nthreads>>> (  device_matrix_A,device_matrix_B, device_matrix_C,m,n,l
							);
	
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


