#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status status;  

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  
  int num_steps = 100000000;
  double sum = 0.0;
  double step = 1.0 / (double) num_steps;
  for (int i = rank; i < num_steps; i += size){
    double x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  double pi_local = step * sum;
  double pi;
  MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - start;
  
  if (rank==0){
    cout << "pi = " << pi << ", elapsed = " << elapsed << endl;
  }

  MPI_Finalize();
  return 0;
}

