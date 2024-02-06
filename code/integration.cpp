#include <omp.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main(){
  double sum = 0.0;
  int num_steps = 1000;
  double step = 1.0 / (double) num_steps;
  
  double elapsed_time = omp_get_wtime();
  for (int i = 0; i < num_steps; i++){
    double x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  double pi = step * sum;
  elapsed_time = omp_get_wtime() - elapsed_time;
  cout << "pi = " << setprecision(7) << pi << " in " << elapsed_time << " secs" << endl;
}
