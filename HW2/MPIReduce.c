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
  int hit_num;
  int hit_total;
  int ierr;
  double l = 1.0;
  int master = 0;
  double pi_estimate;
  int process_num;
  int process_rank;
  int trial_num = 100000;
  int trial_total;
  double start;
  double finish;
  double elapsed;

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


if ( process_rank == master ) 
  {
    printf ( "\n" );
    printf ( "  The number of processes is %d\n", process_num );
    printf ( "\n" );
    printf ( "  Needle length L = %f\n", l );
  }

start = MPI_Wtime();

hit_num = buffon_laplace_simulate ( a, b, l, trial_num );



ierr = MPI_Reduce ( &hit_num, &hit_total, 1, MPI_INT, MPI_SUM, master, 
    MPI_COMM_WORLD );

finish = MPI_Wtime();
elapsed = finish - start;

 if ( process_rank == master )
  {

    trial_total = trial_num * process_num;
    pi_estimate = 2* (double) trial_total / ( double )hit_total;
    printf ( "\n" );
    printf ( 
      "    Trials      Hits        Estimated Pi       \n" );
    printf ( "\n" );
    printf ( "  %8d  %8d    %16.12f \n",
      trial_total, hit_total, pi_estimate );
  }

 if ( process_rank == master )
  {
    printf ( "\n" );
    printf("time Elapsed::%f\n",elapsed);   
    printf ( "BUFFON_LAPLACE - Master process:\n" );
   
 }
 ierr = MPI_Finalize ( );
 return 0;
}
int buffon_laplace_simulate ( double a, double b, double l, int trial_num )

 
{
  double angle;
  int hits;
  int trial;
  double x1=0.0 ;
  double x2;
  double y1=0.0;
  double y2;
  double diff=0.0;
  double delta_x;
  double delta_y;
  double y_l;
  double y_h;
  hits = 0;
  
	for ( trial = 1; trial <= trial_num; trial++ ){


		delta_x =  (double) rand()/(double) RAND_MAX*2;
		delta_y = (double) rand()/ (double) RAND_MAX*2;
		diff = delta_x*delta_x + delta_y*delta_y;
		y1 = (double) rand()/(double) RAND_MAX;
		y2 = y1 + (delta_y/sqrt(diff));
		y_l =y1<y2 ? y1 : y2;
		y_h =  y1>y2 ? y1: y2;
		if(ceil(y_l) == floor(y_h)){
			hits = hits+1;
		}		

	}		
return hits;
}

