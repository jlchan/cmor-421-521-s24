#include <omp.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 8
int main(){
  int num_steps = 100000000;
  double step = 1.0 / (double) num_steps;
  double elapsed_time = omp_get_wtime();
  omp_set_num_threads(NUM_THREADS);
  double pi = 0.0; // this will be shared
#pragma omp parallel
{
    double sum = 0.0; // initialized as a private variable
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (int i = id; i < num_steps; i += nthreads){
      double x = (i + 0.5) * step;
      sum += 4.0 / (1.0 + x * x);
    }
    // only one thread does this per time
    #pragma omp critical 
    {
        pi += step * sum;
    }
}  

  elapsed_time = omp_get_wtime() - elapsed_time;
  cout << "pi = " << pi << " in " << elapsed_time << " secs" << endl;
}
