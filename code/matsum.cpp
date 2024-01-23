#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

double matsum_ij(int n, double* A){
  double val = 0.0;
  for (int i = 0; i < n; ++i){
    for (int j = 0; j < n; ++j){      
      val += A[j + i * n];
    }
  }
  return val;
}

double matsum_ji(int n, double* A){
  double val = 0.0;
  for (int j = 0; j < n; ++j){
    for (int i = 0; i < n; ++i){
      val += A[j + i * n];      
    }
  }
  return val;
}


int main(int argc, char * argv[]){

  int n = atoi(argv[1]);
  cout << "Matrix size n = " << n << endl;
  
  double * A = new double[n * n];

  for (int i = 0; i < n * n; ++i){
    A[i] = 1.0 / (n * n);
  }
  
  int num_trials = 1;
  double val_exact = 1.0;
  double tol = 1e-15 * n * n;
    
  // Measure performance  
  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < num_trials; ++i){
    double val = matsum_ij(n, A);
    if (fabs(val - val_exact) > tol){
      cout << "you did something wrong " << endl;
    }
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> elapsed_ij = (end - start) / num_trials;

  // Measure performance  
  start = high_resolution_clock::now();
  for (int i = 0; i < num_trials; ++i){
    double val = matsum_ji(n, A);
    if (fabs(val - val_exact) > tol){
      cout << "you did something wrong " << endl;
    }    
  }
  end = high_resolution_clock::now();
  duration<double> elapsed_ji = (end - start) / num_trials;

  cout << "ij elapsed time = " << elapsed_ij.count() << endl;
  cout << "ji elapsed time = " << elapsed_ji.count() << endl;  

  delete[] A;
  
  return 0;
}
