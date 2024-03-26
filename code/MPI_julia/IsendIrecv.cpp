#include "mpi.h"
#include <iostream>

using namespace std;

int main(){

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 2;
    int * x = new int [n];
    for (int i = 0; i < n; ++i){
        x[i] = rank;
    }

    MPI_Request requests[2];
    if (rank==0){
        MPI_Isend(x, n, MPI_INT, 1, 0, MPI_COMM_WORLD, &requests[0]);
    } else {
        MPI_Isend(x, n, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(x, n, MPI_INT, (rank - 1), 0, MPI_COMM_WORLD, &requests[1]);
    }

    // rank 0 issues a recv later
    if (rank == 0){
        MPI_Irecv(x, n, MPI_INT, (size - 1), 0, MPI_COMM_WORLD, &requests[1]);
    }
    
    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);

    printf("On rank %d, x[0] = %d\n", rank, x[0]);

    MPI_Finalize();
    return 0;
}

