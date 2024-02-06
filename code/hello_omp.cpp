#include <omp.h>
#include <iostream>

using namespace std;

int main(){

#pragma omp parallel
  {
    //cout << "Running with " << omp_get_num_threads() << " threads" << endl;    
    int tid = omp_get_thread_num();
    cout << "Hello world from thread " << tid << endl;
  }
  return 0;
}
