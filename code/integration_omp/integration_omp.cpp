#include <omp.h>
#include <iostream>

using namespace std;

int main(){
  int num_steps = 100000000;
  double sum = 0.0;
  double step = 1.0 / (double) num_steps;
  omp_set_num_threads(1);
  double elapsed_time = omp_get_wtime();
#pragma omp parallel
  {
    for (int i = 0; i < num_steps; i++){
      double x = (i + 0.5) * step;
      sum = sum + 4.0 / (1.0 + x * x);
    }
  }
  double pi = step * sum;
  elapsed_time = omp_get_wtime() - elapsed_time;
  cout << "pi = " << pi << " in " << elapsed_time << " secs" << endl;
}
