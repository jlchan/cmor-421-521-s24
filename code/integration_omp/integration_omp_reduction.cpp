#include <omp.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 8
int main(){
    int num_steps = 100000000;
    double step = 1.0 / (double) num_steps;
    double elapsed_time = omp_get_wtime();
    omp_set_num_threads(NUM_THREADS);
    double sum = 0.0; // initialize reduction variable as shared
    #pragma omp parallel 
    {        
        #pragma omp for reduction (+:sum)  
            for (int i = 0; i < num_steps; ++i){                
                double x = (i + 0.5) * step; 
                sum += 4.0 / (1.0 + x * x);
            }
    } 
    double pi = sum * step;
    elapsed_time = omp_get_wtime() - elapsed_time;
    cout << "pi = " << pi << " in " << elapsed_time << " secs" << endl;
}
