#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status status;
  int x;
  if (rank==0){
    x = 123;
    MPI_Send(&x, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
  }else if (rank==3){
    MPI_Recv(&x, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }else{
    x = -1;
  }

  // broadcast, reduce, barriers, wait ...
  
  cout << "On rank " << rank << " of " << size << ", x = " << x << endl;
  
  MPI_Finalize();
  return 0;
}
