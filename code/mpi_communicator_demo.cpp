#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // the following assumes size = p x p
  int p = (int) sqrt(size);
  
  // create communicators for each row and column
  int row_color = rank / p;
  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);

  int col_color = rank % p;
  MPI_Comm col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);

  cout << "on rank " << rank;
  cout << "(row, col) = " << row_color << ", " << col_color << endl;
  
  int x = rank;
  MPI_Bcast(&x, 1, MPI_INT, 0, row_comm);
    
  cout << "After bcast, on rank " << rank << ", x = " << x << endl;

  MPI_Finalize();

  return 0;
}
