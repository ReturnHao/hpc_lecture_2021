//
//  Matrix_mult_OpenMP.cpp
//  Parallel Matrix Multiplication using OpenMP
//
//  Author: Fang Hao
//
//  Compile with -fopenmp -std=c++11 flag
//  ./a.out
//
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main(int argc, char **argv)
{
    // Generate Matrix
    const int N = 256;
    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[N * i + j] = drand48();
            B[N * i + j] = drand48();
        }
    }
    
    // Record the time-stamp before parallel process start
    auto toc = chrono::steady_clock::now();
    
    // Parallel Matrix Multiplication
    int i, j, k;

# pragma omp parallel shared (A, B, C) private (i, j, k)
{
# pragma omp for
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[N * i + j] += A[N * i + k] * B[N * k + j];
}
  
    // Record the time-stamp after parallel process end & Calculate the total time
    auto tic = chrono::steady_clock::now();
    double time = chrono::duration<double>(tic - toc).count();
    
    // Check Error
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[N * i + j] -= A[N * i + k] * B[N * k + j];
    
    double err = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            err += fabs(C[N * i + j]);
    
    // Output Result
    printf("N    : %d\n", N);
    printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
    printf("error: %lf\n", err / N / N);
}
