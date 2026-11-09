/* Task 4 — test_gemm: verify CBLAS gemm_xt matches model.cu numerics.
 *
 * Uses the CblasColMajor mirror of the cuBLAS call:
 *   cublasSgemm(h, OP_T, OP_N, out, B, in, &a, W, in, X, in, &b, Y, out);
 * becomes
 *   cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, out, B, in,
 *               1.f, W, in, X, in, 0.f, Y, out);
 *
 * Build:  make tools/test_gemm
 * Run:    ./tools/test_gemm   (expected: "test_gemm ok")
 */
#ifdef ANN_BLAS_BUILTIN
#include "cblas_builtin.h"
#else
#include <cblas.h>
#endif
#include <math.h>
#include <stdio.h>

/* Y(B,out) = X(B,in) @ W(out,in)^T — same math as model.cu gemm_xt. */
static void gemm_xt(int B, int out, int in, const float *X, const float *W,
                     float *Y)
{
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, out, B, in, 1.f, W, in,
                X, in, 0.f, Y, out);
}

int main(void)
{
    float X[2 * 3] = {1, 2, 3, 4, 5, 6};  /* B=2, in=3 (row-major) */
    float W[2 * 3] = {1, 0, 0, 0, 1, 0};  /* out=2, in=3 (row-major) */
    float Y[2 * 2];
    gemm_xt(2, 2, 3, X, W, Y);
    /* expect Y row0 = (1,2), row1 = (4,5) */
    if (fabsf(Y[0] - 1.f) > 1e-5f || fabsf(Y[1] - 2.f) > 1e-5f ||
        fabsf(Y[2] - 4.f) > 1e-5f || fabsf(Y[3] - 5.f) > 1e-5f) {
        fprintf(stderr, "gemm_xt mismatch %g %g %g %g\n", Y[0], Y[1], Y[2], Y[3]);
        return 1;
    }
    printf("test_gemm ok\n");
    return 0;
}
