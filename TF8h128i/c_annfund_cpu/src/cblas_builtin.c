#include "cblas_builtin.h"

#include <stdio.h>
#include <stdlib.h>

/* Naive ColMajor SGEMM: C = alpha*op(A)*op(B) + beta*C
 * op(A) is MxK, op(B) is KxN, C is MxN.
 * Only the layouts/ops used by model.c are required; others abort. */
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc)
{
    if (Order != CblasColMajor) {
        fprintf(stderr, "cblas_builtin: only CblasColMajor supported\n");
        abort();
    }
    if (TransA == CblasConjTrans || TransB == CblasConjTrans) {
        fprintf(stderr, "cblas_builtin: ConjTrans not supported\n");
        abort();
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                float a = (TransA == CblasNoTrans) ? A[i + k * lda] : A[k + i * lda];
                float b = (TransB == CblasNoTrans) ? B[k + j * ldb] : B[j + k * ldb];
                sum += a * b;
            }
            float *cij = &C[i + j * ldc];
            if (beta == 0.f) {
                *cij = alpha * sum;
            } else {
                *cij = alpha * sum + beta * (*cij);
            }
        }
    }
}
