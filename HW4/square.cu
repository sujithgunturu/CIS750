#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


#include <cuda.h>

#define N        10000000		// total number of items in vectors
#define nthreads 4	   // total number of threads in a block


__global__ void square(int n, int *vect1, int *vect2, int *sum)
{
	int threadID;
	threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID < n)
	sum[threadID] = vect1[threadID] * vect1[threadID] + vect2[threadID] * vect2[threadID];
}

int main()
{			
	srand(time(NULL));	
	int *vect1_h, *vect2_h, *sum_h;
	int *vect1_d, *vect2_d, *sum_d;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	vect1_h = (int*)malloc( N* sizeof(int));
	vect2_h = (int*)malloc( N* sizeof(int));
	sum_h   = (int*)malloc( N* sizeof(int));
	
	cudaMalloc((void**)&vect1_d, N * sizeof(int));
	cudaMalloc((void**)&vect2_d, N * sizeof(int));
	cudaMalloc((void**)&sum_d,   N * sizeof(int));

	for(int i = 0; i < N; i++)
	{	
		vect1_h[i] = rand()%10;
		vect2_h[i] = rand()%10;
	}

	cudaMemcpy(vect1_d, vect1_h, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vect2_d, vect2_h, N * sizeof(int), cudaMemcpyHostToDevice);

	
	int nblocks = (N + nthreads - 1)/nthreads; 
		cudaEventRecord(start);
	square<<<nblocks,nthreads>>>(N, vect1_d, vect2_d, sum_d);
    	cudaEventRecord(stop);   	
	cudaMemcpy(sum_h, sum_d, N * sizeof(int), cudaMemcpyDeviceToHost);
	 
	 
	printf("Vector1: \n");	
	for(int i = 0; i < N; ++i)
		printf("  %d", vect1_h[i]);
	
	printf("\nVector2: \n");	
	for(int i = 0; i < N; ++i)
		printf("  %d", vect2_h[i]);

	
	printf("\nThe sum of squares of the vecors are is: \n");	
	for(int i = 0; i < N; ++i)
		printf("  %d", sum_h[i]);
	printf("\n");
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("elaspsed = %f ms", milliseconds);
	
	cudaFree(vect1_d);
	cudaFree(vect2_d);
	cudaFree(sum_d);
	
	free(vect1_h);
	free(vect2_h);
	free(sum_h);
}
