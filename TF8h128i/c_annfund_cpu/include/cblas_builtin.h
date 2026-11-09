#ifndef ANN_CBLAS_BUILTIN_H
#define ANN_CBLAS_BUILTIN_H

/* Minimal CBLAS subset for ColMajor sgemm (OpenBLAS-free fallback). */

enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc);

#endif
