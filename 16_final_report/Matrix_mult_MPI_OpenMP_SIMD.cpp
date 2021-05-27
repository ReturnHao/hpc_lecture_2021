//
//  Matrix_mult_MPI_OpenMP_SIMD.cpp
//  Parallel Matrix Multiplication using Hybird on MPI and OpenMP
//
//  Author: Fang Hao
//
//  Compile with -fopenmp -std=c++11 flag
//  mpirun -np 4 ./a.out
//
#include <bits/stdc++.h>
#include <immintrin.h>
#include <mpi.h>
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
    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0);
    // vector<float> subA(N * N / size);
    // vector<float> subB(N * N / size);
    // vector<float> subC(N * N / size, 0);
    float subA[N * N / size];
    float subB[N * N / size];
    float subC[N * N / size];
    
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
    for (int i = 0; i < N * N; i++) subC[i] = 0;
    int recv_from = (rank + 1) % size;
    int send_to = (rank - 1 + size) % size;

    // Initialize simd Block
    // __m256 AVec, BVec, CVec;
    float Temp[8];
    int n_chunks = 4;
    
    // Parallel Matrix Multiplication
    double comp_time = 0, comm_time = 0;
    for (int irank = 0; irank < size; irank++)
    {
        // Record the time-stamp before sub parallel process start
        auto tic = chrono::steady_clock::now();
        
        offset = N / size * ((rank + irank) % size);
        /*
        int i, j, k;
# pragma omp parallel shared (subA, subB, subC, size, offset) private (i, j, k)
{
# pragma omp for
        for (i = 0; i < N / size; i++)
            for (j = 0; j < N / size; j++)
                for (k = 0; k < N; k++)
                    subC[N * i + j + offset] += subA[N * i + k] * subB[N / size * k + j];
}
        */
        int chunk, i, j, k;
# pragma omp parallel for
        for (chunk = 0; chunk < 4; chunk++)
            for (i = chunk * (N / size / n_chunks); i < (chunk + 1) * (N / size / n_chunks); i++)
            {
                for (j = 0; j < N / size; j++)
                {
                    for (k = 0; k < 8; k++) Temp[k] = 0.0f;
                    __m256 CVec = _mm256_loadu_ps(Temp);
                    for (k = 0; k < N; k += 8)
                    {
                        // load
                        __m256 AVec = _mm256_loadu_ps(&subA[N * i + k]);
                        __m256 BVec = _mm256_loadu_ps(&subB[N / size * k + j]);
                        
                        // fused multiply and add
                        CVec = _mm256_fmadd_ps(AVec, BVec, CVec);
                    }
                    
                    _mm256_storeu_ps(Temp, CVec);
                    for (k = 0; k < 8; k++) subC[N * i + j + offset] += Temp[k];
                }
            }
            
        // Record the time-stamp after sub calculation process end
        auto toc = chrono::steady_clock::now();
        comp_time += chrono::duration<double>(toc - tic).count();
        
        // Send & Receive Buffer
        MPI_Send(&subB[0], N * N / size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
        MPI_Recv(&subB[0], N * N / size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Record the time-stamp after whole process end
        tic = chrono::steady_clock::now();
        comm_time += chrono::duration<double>(tic - toc).count();
    }
    // Allgather result data
    MPI_Allgather(&subC[0], N * N / size, MPI_FLOAT, &C[0], N * N / size, MPI_FLOAT, MPI_COMM_WORLD);
  
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