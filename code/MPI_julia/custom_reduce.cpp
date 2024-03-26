#include "mpi.h"
#include <stdio.h>
  
// compute y += x
void add_vectors(int *x, int *y, int *n, MPI_Datatype *dtype){
    for (int i=0; i < (*n); i++){
        y[i] += x[i];
    }
}
 
int main(void){
    int rank, size, i;
    MPI_Op op;
 
    MPI_Init(NULL, NULL);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    int n = 3;
    int * x = new int[n];
    int * y = new int[n];
    for (int i = 0; i < n; ++i){
        x[i] = rank;
        y[i] = 0;
    }
    MPI_Op_create((MPI_User_function *)add_vectors, 1, &op); // 1 = commutes or not

    MPI_Reduce(x, y, n, MPI_INT, op, 0, MPI_COMM_WORLD);
    MPI_Op_free(&op);
    if (rank==0){
        for (int i = 0; i < n; ++i){
            printf("y[%d] = %d\n", i, y[i]);
        }
    }

    MPI_Finalize();
    return 0;
}