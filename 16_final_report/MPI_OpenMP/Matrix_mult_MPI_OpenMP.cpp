//
//  Matrix_mult_MPI_OpenMP.cpp
//  Parallel Matrix Multiplication using Hybird on MPI and OpenMP
//
//  Author: Fang Hao
//
//  Compile with -fopenmp -std=c++11 flag
//  mpirun -np 4 ./a.out
//
#include <mpi.h>
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv)
{
    // Initialize Process Number & Id
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate Matrix
    const int N = 256;
    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N, 0);
    vector<double> subA(N * N / size);
    vector<double> subB(N * N / size);
    vector<double> subC(N * N / size, 0);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[N * i + j] = drand48();
            B[N * i + j] = drand48();
        }
    }
    
    // Initialize Partition Index & Matrices and Parallel rank
    int offset = N / size * rank;
    for (int i = 0; i < N / size; i++)
        for (int j = 0; j < N; j++)
            subA[N * i + j] = A[N * (i + offset) + j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N / size; j++)
            subB[N / size * i + j] = B[N * i + j + offset];
    int recv_from = (rank + 1) % size;
    int send_to = (rank - 1 + size) % size;

    // Parallel Matrix Multiplication
    double comp_time = 0, comm_time = 0;
    for (int irank = 0; irank < size; irank++)
    {
        // Record the time-stamp before sub parallel process start
        auto tic = chrono::steady_clock::now();
        
        offset = N / size * ((rank + irank) % size);
        int i, j, k;
# pragma omp parallel shared (subA, subB, subC, size, offset) private (i, j, k)
{
# pragma omp for
        for (i = 0; i < N / size; i++)
            for (j = 0; j < N / size; j++)
                for (k = 0; k < N; k++)
                    subC[N * i + j + offset] += subA[N * i + k] * subB[N / size * k + j];
}
        
        // Record the time-stamp after sub calculation process end
        auto toc = chrono::steady_clock::now();
        comp_time += chrono::duration<double>(toc - tic).count();
        
        // Send & Receive Buffer
        MPI_Send(&subB[0], N * N / size, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);
        MPI_Recv(&subB[0], N * N / size, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Record the time-stamp after whole process end
        tic = chrono::steady_clock::now();
        comm_time += chrono::duration<double>(tic - toc).count();
    }
    // Allgather result data
    MPI_Allgather(&subC[0], N * N / size, MPI_DOUBLE, &C[0], N * N / size, MPI_DOUBLE, MPI_COMM_WORLD);
  
    // Error check
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[N * i + j] -= A[N * i + k] * B[N * k + j];
    
    double err = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            err += fabs(C[N * i + j]);
    
    // Output Result
    if (rank == 0)
    {
        double time = comp_time + comm_time;
        printf("N    : %d\n", N);
        printf("comp : %lf s\n", comp_time);
        printf("comm : %lf s\n", comm_time);
        printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
        printf("error: %lf\n", err / N / N);
    }
    
    // Finalization
    MPI_Finalize();
}
