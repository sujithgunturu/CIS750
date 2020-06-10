# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include "mpi.h"

int main ( int argc, char *argv[] );
int buffon_laplace_simulate ( double a, double b, double l, int trial_num );
int main ( int argc, char *argv[] )
{
  double a = 1.0;
  double b = 1.0;
  int hit_num = 0; 
  int hit_total;
  int ierr;
  double l = 1.0;
  int master = 0;
  double pi_estimate;
  int process_num;
  int process_rank;
  double random_value;
  int seed;
  int trial_num = 1000000;
  int trial_total;
  double start;
  double finish;
  double elapsed;
  int tmp;
  MPI_Status stat;

  
  ierr = MPI_Init ( &argc, &argv );

  if ( ierr != 0 )
  {
    printf ( "\n" );
    printf ( "BUFFON_LAPLACE: Warning!\n" );
    printf ( "  MPI_INIT returns IERR = %d\n", ierr );
    ierr = MPI_Finalize ( );
    exit ( 1 );
  }
ierr = MPI_Comm_size ( MPI_COMM_WORLD, &process_num );
ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &process_rank );
start = MPI_Wtime();
hit_num = buffon_laplace_simulate ( a, b, l, trial_num );
finish = MPI_Wtime();
elapsed = finish - start;
MPI_Send (&hit_num,1, MPI_INT, 0, 0, MPI_COMM_WORLD  );


if ( process_rank == master )
  {
for(int i =0; i < process_num ; i++){
 MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
            hit_total += tmp; 
}
    
trial_total = trial_num * process_num;
pi_estimate = 2 * (double)trial_total/(double)hit_total; 
printf ( "\n" );
printf ( 
      "    Trials      Hits          Estimated Pi        \n" );
    printf ( "\n" );
    printf ( "  %8d  %8d   %16.12f  \n",
      trial_total, hit_total, pi_estimate);
    printf("time Elapsed::%f\n",elapsed);   

  }

 ierr = MPI_Finalize ( );

return 0;
}
int buffon_laplace_simulate ( double a, double b, double l, int trial_num )
   
{
  double angle;
  int hits;
  int trial;
  double x1;
  double x2;
  double y1;
  double y2;
  double diff;
  double delta_x;
  double delta_y;
  double y_l;
  double y_h;
  hits = 0;
 
	for ( trial = 1; trial <= trial_num; trial++ )
	{			
    while(x1*x1 +y1*y1 > 1){
	x1 = (double) rand();
	y1 = (double) rand();
	}
	angle = atan(y1/x1);
	diff =(double) rand()/RAND_MAX * 0.7877 ;	
	if(2*diff <=  sin(angle)){
	hits = hits+1;	
}
	}



return hits;
}







