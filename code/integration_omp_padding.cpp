#include <omp.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 8
#define PAD 8
int main(){
  double sum[NUM_THREADS][PAD];
  for (int id = 0; id < NUM_THREADS; ++id){
    sum[id][0] = 0.0;
  }
  int num_steps = 100000000;
  double step = 1.0 / (double) num_steps;
  double elapsed_time = omp_get_wtime();
  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (int i = id; i < num_steps; i += nthreads){
      double x = (i + 0.5) * step;
      sum[id][0] = sum[id][0] + 4.0 / (1.0 + x * x);
    }
  }
  double pi = 0.0;
  for (int id = 0; id < NUM_THREADS; ++id){
    pi += step * sum[id][0];
  }
  elapsed_time = omp_get_wtime() - elapsed_time;
  cout << "pi = " << pi << " in " << elapsed_time << " secs" << endl;
}
