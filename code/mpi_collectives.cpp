#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Bcast 
  int x;
  if (rank==0){
    x = 1;
  }
  MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  cout << "On rank " << rank << "/" << size-1 << ", x = " << x << endl;

  // reduce  
  int sum;
  MPI_Reduce(&x, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0){
    cout << "On root rank, sum = " << sum << endl;
  }


  MPI_Finalize();

  return 0;
}
