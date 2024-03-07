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

  // gather/scatter
  int * xsend = (rank == 0) ? new int [2 * size] : NULL;
  int * xrecv = new int [2];
  if (rank==0){
    for (int i = 0; i < 2 * size; ++i){
      xsend[i] = i;
    }
  }
  MPI_Scatter(xsend, 2, MPI_INT,
	      xrecv, 2, MPI_INT, 0,
	      MPI_COMM_WORLD);

  cout << "on rank " << rank << ", xrecv = ";
  cout << xrecv[0] << ", " << xrecv[1] << endl;


  MPI_Finalize();

  return 0;
}
