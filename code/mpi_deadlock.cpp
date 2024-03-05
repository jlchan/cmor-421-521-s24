#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 1;
  int * x = new int [n];

  if (rank==0){
    MPI_Send(x, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(x, n, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }else if (rank==1){
    MPI_Send(x, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(x, n, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /*
  MPI_Request request;   
  if (rank==0){
    MPI_Isend(x, n, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    MPI_Irecv(x, n, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }else if (rank==1){
    MPI_Isend(x, n, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Irecv(x, n, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  */
  
  MPI_Finalize();
  return 0;
}

