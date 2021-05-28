//
//  Matrix_mult_MPI_OpenMP_SIMD.cpp
//  Parallel Matrix Multiplication using Hybird on MPI and OpenMP with SIMD
//
//  Author: Fang Hao
//
//  Compile with:
//  mpicxx Matrix_mult_MPI_OpenMP_SIMD.cpp -fopenmp -march=native -O3 -std=c++11
//  mpirun -np 4 ./a.out
//
#include <immintrin.h>
#include <bits/stdc++.h>
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
    vector<float> subA(N * N / size);
    vector<float> subB(N * N / size);
    vector<float> subC(N * N / size, 0);
    
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

    // Initialize OpenMP
    // int thread_count = omp_get_num_threads();
    
    // Initialize SIMD Blocking
    float columnSect[N];
    int n_chunks = N / 64;
    
    // Parallel Matrix Multiplication
    double comp_time = 0, comm_time = 0;
    for (int irank = 0; irank < size; irank++)
    {
        // Record the time-stamp before sub parallel process start
        auto tic = chrono::steady_clock::now();
        
        offset = N / size * ((rank + irank) % size);
        int chunk, i, j, k;
# pragma omp parallel shared (subA, subB, subC, size, offset) private (chunk, i, j, k, columnSect)
{        
# pragma omp for
        for (chunk = 0; chunk < n_chunks; chunk++)
        {
            for (i = chunk * (N / size / n_chunks); i < (chunk + 1) * (N / size / n_chunks); i++)
            {
                for (j = 0; j < N / size; j++)
                {
                    // Acquire column data
                    for (k = 0; k < N; k++)
                    {
                        columnSect[k] = subB[N / size * k + j];
                    }
                    
                    __m128 vc = _mm_set_ps1(0.0f);
                    for (k = 0; k < N; k += 4)
                    {
                        // Load row & column data
                        __m128 va = _mm_load_ps(&subA[N * i + k]);
                        __m128 vb = _mm_load_ps(&columnSect[k]);
                        
                        // Multiply & add
                        __m128 vres = _mm_mul_ps(va, vb);
                        vc = _mm_add_ps(vc, vres);
                    }
                    // Reduce to a single float
                    vc = _mm_hadd_ps(vc, vc);
                    vc = _mm_hadd_ps(vc, vc);
                    
                    // Store result
                    subC[N * i + j + offset] = _mm_cvtss_f32(vc);
                }
            }
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
