#include <stdio.h>
#include <omp.h>
#include <math.h>


int main()
{
    double arr[30];
    omp_set_num_threads(4);
    double max_value=0.0, min_value = 0.0;
    int i;
    for( i=0; i<30; i++){
	arr[i] = mrand48();

    }
    for(i = 0; i< 30; i++){
	printf("%f\n", arr[i]);
    }
    #pragma omp parallel for reduction(max : max_value) reduction(min:min_value)
    	for( i=0;i<30; i++){
        	if(arr[i] > max_value){
            		max_value = arr[i];  
        	}
		if(arr[i] < min_value){
	    		min_value = arr[i];
		}
    	}
  
    printf("\nmaximum value  = %f", max_value);
    printf("\n minimum value = %f", min_value);		
}