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
  int tag = 0;
  if (rank==0){
    x = 123;
    MPI_Send(&x, 1, MPI_INT, size-1, tag, MPI_COMM_WORLD);
  }else if (rank==size-1){
    MPI_Recv(&x, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
  }

  cout << "On rank " << rank << " of " << size << ", x = " << x << endl;
  
  MPI_Finalize();
  return 0;
}
