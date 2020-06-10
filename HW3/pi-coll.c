# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <omp.h>

int main ( int argc, char *argv[] );
int main ( int argc, char *argv[] ){
	double a = 1.0;
	double b = 1.0;
	int hit_num = 0; 
	int hit_total;
 	double l = 1.0;
	int master = 0;
	double pi_estimate;
	double random_value;
  	int trial_num = 100000;
  	int trial_total;
  	double start;
  	double finish;
  	double elapsed;
 	int NUM_THREADS = 8;
  	int tid;
	double angle;
    	int trial;
  	double x1;
  	double y1;
    	double diff;
  	int hits = 0;
	start = omp_get_wtime();
	omp_set_num_threads(NUM_THREADS);
	# pragma omp parallel for reduction(+:hits)
	for ( trial = 1; trial <= trial_num; trial++ ){	
		x1 = drand48();
   		y1 = drand48();
    		while(x1*x1 +y1*y1 > 1){
			x1 = drand48();
   			y1 = drand48();

		}
		angle = atan(y1/x1);
		diff = drand48()*0.5;	
		if(diff <=  sin(angle) * 0.5){
			hits = hits+1;	
		}
	}

	finish = omp_get_wtime();
	elapsed = finish - start;

		trial_total = trial_num; 
		pi_estimate = 2 * (double)trial_total/(double)hits; 
		printf ( "\n" );
		printf ( "    Trials      Hits          Estimated Pi       Time \n" );
    		printf ( "\n" );
    		printf ( "  %8d  %8d   %16.12f  %f\n",
      			trial_total, hits, pi_estimate, elapsed);
    		

  	

return 0;
}











