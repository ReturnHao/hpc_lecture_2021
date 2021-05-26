#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <vector>
#include <immintrin.h>
using namespace std;
typedef vector<vector<float>> matrix;

void matmult(matrix &A, matrix &B, matrix &C, int N) {
float a[N*N],b[N*N],c[N*N];
for (int i=0;i<N;i++){
 for(int j=0;j<N;j++){
  a[i*N+j]=A[i][j];
  b[i*N+j]=B[i][j];
  c[i*N+j]=0;
}
}
 for(int i=0; i<N; i++){
  for(int j = 0; j < N; j++){
   for(int k=0;k<N;k+=8){
     __m256 Avec = _mm256_load_ps(a+i*N+k);
     __m256 Bvec = _mm256_load_ps(b+k*N+j);
     __m256 Cvec = _mm256_load_ps(c+i*N+j);
     Cvec = _mm256_fmadd_ps(Avec, Bvec, Cvec);
      _mm256_store_ps(c+i*N+j, Cvec);
  }
 }
}
for (int i=0;i<N;i++){
 for (int j=0;j<N;j++){

  C[i][j]=c[i*N+j];
  }
 }
}
int main(int argc, char **argv) {
  const int N = 4096;
  matrix A(N,vector<float>(N));
  matrix B(N,vector<float>(N));
  matrix C(N,vector<float>(N));
  matmult(A,B,C,N);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  auto tic = chrono::steady_clock::now();
  matmult(A,B,C,N);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[i][j] -= A[i][k] * B[k][j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[i][j]);
  printf("error: %lf\n",err/N/N);
}
