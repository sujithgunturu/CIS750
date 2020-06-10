#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include <cuda.h>
#include <curand_kernel.h>

#define N        100		// total number of items in vectors
#define nthreads 4	   // total number of threads in a block



__global__ void estimatepi(int n, int *sum)

{
	__shared__ int counter[nthreads];
	int threadID;
	threadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int seed = threadID;
	curandState s;
	curand_init(seed, 0, 0, &s);
	if(threadID < n){
		double x, y, diff, angle;
		int t;
		counter[threadIdx.x] = 0;
		for (t = 0; t<n; t++){
			x = curand_uniform(&s); //curand
			y = curand_uniform(&s); //curand
			while(x*x + y*y > 1){
				x = curand_uniform(&s); //curand
				y = curand_uniform(&s); //rand
			}

			angle = atan2 ( y, x ); //use inverse tan;
			diff = curand_uniform(&s);
			if(diff <= sin (angle) *2){
				counter[threadIdx.x] = counter[threadIdx.x] + 1;
			}
		}

		if(threadIdx.x == 0){
			sum[blockIdx.x] = 0;
			for(int i=0; i<nthreads; i++)	{
				sum[blockIdx.x] = sum[blockIdx.x] + counter[i];
			}
		}
	}
}

int main()
{	
		
		srand(time(NULL));	
		int  *sum_h;
		int *sum_d;
		
	
		sum_h   = (int*)malloc( N* sizeof(int));
	
		cudaMalloc((void**)&sum_d,   N * sizeof(int));


	
		int nblocks = (N + nthreads - 1)/nthreads; 
		estimatepi<<<nblocks,nthreads>>>(N,sum_d);

    
   		cudaMemcpy(sum_h, sum_d, N * sizeof(int), cudaMemcpyDeviceToHost);

	 
		int success = 0;
		for(int i = 0; i < nblocks; i++){
		success = sum_h[i] + success; 
}
		printf("trials === %d", N * nblocks * nthreads );		
		printf("  success === %d\n", success);
		double  pi_estimate = 2 * N * nthreads * nblocks/( double )success;
		printf("pi_estimate == %f", pi_estimate);
		

		printf("\n");
		cudaFree(sum_d);
		free(sum_h);
}
 	