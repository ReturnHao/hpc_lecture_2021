#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

#define N 512

float matrix_a[N*N];
float matrix_b[N*N];
float c[N*N];
float result[N][N];

/*void chunked_mm(int chunk, int n_chunks) {
    __m256 va, vb, vc;
    for (int i = chunk*(N/n_chunks); i < (chunk+1)*(N/n_chunks); i++) {
        for (int j = 0; j < N; j++) {
            float buffer[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            vc = _mm256_loadu_ps(buffer);
            for (int k = 0; k < N; k += 8) {
                // load
                va = _mm256_loadu_ps(matrix_a+(i*N)+k); // matrix_a[i][k]
                vb = _mm256_loadu_ps(matrix_b+(j*N)+k); // matrix_b[j][k]

                // fused multiply and add
                vc = _mm256_fmadd_ps(va, vb, vc);
            }
            //vc = _mm256_hadd_ps(vc, vc);
            _mm256_storeu_ps(buffer, vc);
            result[i][j] = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
            //result[i][j] = buffer[0] + buffer[2] + buffer[4] + buffer[6];
        }
    }
}*/

void chunked_mm()
{
    for(int i=0;i<N;i+=1){

          for(int j=0;j<N;j+=4){

            for(int k=0;k<N;k+=4){

              __m128 result = _mm_load_ps(&result[i][j]);

              __m128 a_line  = _mm_load_ps(matrix_a+ i * N + k]);

              __m128 b_line0 = _mm_load_ps(matrix_b + k * N + j);

              __m128 b_line1 = _mm_loadu_ps(matrix_b + k * N + j + 1);

              __m128 b_line2 = _mm_loadu_ps(matrix_b + k * N + j + 2);

              __m128 b_line3 = _mm_loadu_ps(matrix_b + k * N + j + 3);

             result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a_line, a_line, 0x00), b_line0));
             result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a_line, a_line, 0x55), b_line1));
             result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a_line, a_line, 0xaa), b_line2));
             result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a_line, a_line, 0xff), b_line3));
             _mm_store_ps(&result[i][j],result);
            }
          }
        }
}

int main(int argc, char **argv) {
    // initialize matrix_a and matrix_b
    // matrix_a = malloc(N*N*sizeof(float));
    // matrix_b = malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++) {
        *(matrix_a+i) = drand48();
        *(matrix_b+i) = drand48();
    }
    // initialize result matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0.0f;
        }
    }

    /*
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        chunked_mm(i, 4);
    }
    */
    chunked_mm();
    
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
          for (int k=0; k<N; k++)
            result[i][j] -= matrix_a[N * i + k] * matrix_b[N * j + k];
    
    double err = 0;
      for (int i=0; i < N; i++)
        for (int j=0; j < N; j++)
          err += fabs(result[i][j]);
    
    printf("error: %lf\n", err / N / N);
    
    return 0;
}
