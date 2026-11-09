#include "model.h"
#include "weights.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "win_compat.h"

#define ANN_UDP_HOST "8.138.255.148"
#define ANN_UDP_PORT 48888

static void udp_log_loss(const char *code, float avg)
{
    char msg[64];
    int n = snprintf(msg, sizeof(msg), "%s %.6f", code, avg);
    if (n <= 0) {
        return;
    }
    (void)ann_udp_send(ANN_UDP_HOST, (unsigned)ANN_UDP_PORT, msg, n);
}

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _e = (stmt);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(_e));                                   \
            return ANN_ERR;                                                    \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(stmt)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (stmt);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS %s:%d status=%d\n", __FILE__, __LINE__,    \
                    (int)_s);                                                  \
            return ANN_ERR;                                                    \
        }                                                                      \
    } while (0)

static void invalidate_static_ws(void);

int cuda_runtime_ok(void)
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

int cuda_harden_reset(const char *why)
{
    cudaError_t sync_e = cudaDeviceSynchronize();
    cudaError_t last_e = cudaGetLastError();
    fprintf(stderr, "cuda_harden (%s): sync=%s last=%s\n", why ? why : "",
            cudaGetErrorString(sync_e), cudaGetErrorString(last_e));
    if (sync_e != cudaSuccess || last_e != cudaSuccess) {
        /* launch failure 之后 context 已废，不能再 cublasDestroy / cudaFree。
         * cudaDeviceReset 拆掉 context，但不会自动选中设备；
         * 不调用 cudaSetDevice 时，下一次 cublasCreate 会失败。 */
        cudaDeviceReset();
        cudaError_t set_e = cudaSetDevice(0);
        if (set_e != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice after reset: %s\n",
                    cudaGetErrorString(set_e));
        }
        invalidate_static_ws();
        fprintf(stderr, "cudaDeviceReset after error; recreate models before train\n");
        return ANN_ERR;
    }
    return ANN_OK;
}

static float *dalloc(size_t n)
{
    float *p = NULL;
    if (n == 0) {
        return NULL;
    }
    if (cudaMalloc((void **)&p, n * sizeof(float)) != cudaSuccess) {
        return NULL;
    }
    cudaMemset(p, 0, n * sizeof(float));
    return p;
}

static void dfree(float *p)
{
    if (p) {
        cudaFree(p);
    }
}

static int blocks(int n) { return (n + 255) / 256; }

/* Y(B,out) = X(B,in) @ W(out,in)^T */
static int gemm_xt(cublasHandle_t h, int B, int out, int in, const float *X,
                   const float *W, float *Y)
{
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, B, in, &alpha, W,
                             in, X, in, &beta, Y, out));
    return ANN_OK;
}

static int gemm_dx(cublasHandle_t h, int B, int out, int in, const float *dY,
                   const float *W, float *dX, float beta)
{
    const float alpha = 1.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, in, B, out, &alpha, W,
                             in, dY, out, &beta, dX, in));
    return ANN_OK;
}

static int gemm_dw(cublasHandle_t h, int B, int out, int in, const float *dY,
                   const float *X, float *dW)
{
    const float alpha = 1.f, beta = 1.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, in, out, B, &alpha, X,
                             in, dY, out, &beta, dW, in));
    return ANN_OK;
}

__global__ void k_add_bias(float *y, const float *b, int B, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * D) {
        y[i] += b[i % D];
    }
}

__global__ void k_bias_grad(const float *dy, float *db, int B, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) {
        return;
    }
    float s = 0;
    for (int i = 0; i < B; i++) {
        s += dy[i * D + d];
    }
    db[d] += s;
}

__global__ void k_add(float *y, const float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

__global__ void k_add_pe(float *y, const float *x, const float *pe, int B, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * H) {
        y[i] = x[i] + pe[i % H];
    }
}

__global__ void k_relu(float *y, const float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] > 0.f ? x[i] : 0.f;
    }
}

__global__ void k_relu_bwd(float *dx, const float *x, const float *dy, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = x[i] > 0.f ? dy[i] : 0.f;
    }
}

__global__ void k_l1_bwd(const float *pred, const float *tgt, float *dpred, int n,
                         float *loss_buf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float loc = 0;
    if (i < n) {
        float d = pred[i] - tgt[i];
        loc = fabsf(d);
        dpred[i] = (d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f)) / (float)n;
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = (i < n) ? loc : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(loss_buf, sh[0]);
    }
}

__global__ void k_ln_fwd(const float *x, float *y, const float *w, const float *b,
                         float *mean, float *rstd, int B, int H, float eps)
{
    int row = blockIdx.x;
    if (row >= B) {
        return;
    }
    const float *xr = x + row * H;
    float m = 0;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        m += xr[i];
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    m = sh[0] / (float)H;
    if (threadIdx.x == 0) {
        mean[row] = m;
    }
    __syncthreads();
    float v = 0;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float d = xr[i] - m;
        v += d * d;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    float rs = rsqrtf(sh[0] / (float)H + eps);
    if (threadIdx.x == 0) {
        rstd[row] = rs;
    }
    __syncthreads();
    float *yr = y + row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        yr[i] = (xr[i] - m) * rs * w[i] + b[i];
    }
}

__global__ void k_ln_bwd(const float *x, const float *dy, const float *w,
                         const float *mean, const float *rstd, float *dx,
                         float *dw, float *db, int B, int H)
{
    int row = blockIdx.x;
    if (row >= B) {
        return;
    }
    const float *xr = x + row * H;
    const float *dyr = dy + row * H;
    float m = mean[row], rs = rstd[row];
    float dnorm_sum = 0, dnorm_xsum = 0;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xhat = (xr[i] - m) * rs;
        float dyw = dyr[i] * w[i];
        dnorm_sum += dyw;
        dnorm_xsum += dyw * xhat;
        atomicAdd(dw + i, dyr[i] * xhat);
        atomicAdd(db + i, dyr[i]);
    }
    __shared__ float sh0[256], sh1[256];
    sh0[threadIdx.x] = dnorm_sum;
    sh1[threadIdx.x] = dnorm_xsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh0[threadIdx.x] += sh0[threadIdx.x + s];
            sh1[threadIdx.x] += sh1[threadIdx.x + s];
        }
        __syncthreads();
    }
    dnorm_sum = sh0[0];
    dnorm_xsum = sh1[0];
    float *dxr = dx + row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xhat = (xr[i] - m) * rs;
        float dyw = dyr[i] * w[i];
        dxr[i] = rs * (dyw - dnorm_sum / (float)H - xhat * dnorm_xsum / (float)H);
    }
}

__global__ void k_dropout(float *y, uint8_t *mask, const float *x, int n,
                          float drop_p, uint32_t seed, int train)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (!train || drop_p <= 0.f) {
        y[i] = x[i];
        if (mask) {
            mask[i] = 1;
        }
        return;
    }
    uint32_t h = (uint32_t)i * 747796405u + seed;
    h ^= h >> 17;
    h *= 2246822519u;
    float u = (h >> 8) * (1.f / 16777216.f);
    uint8_t keep = u >= drop_p;
    mask[i] = keep;
    y[i] = keep ? x[i] / (1.f - drop_p) : 0.f;
}

__global__ void k_dropout_bwd(float *dx, const uint8_t *mask, const float *dy,
                              int n, float drop_p, int train)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (!train || drop_p <= 0.f) {
        dx[i] = dy[i];
        return;
    }
    dx[i] = mask[i] ? dy[i] / (1.f - drop_p) : 0.f;
}

__global__ void k_axpy(float *y, const float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += x[i];
    }
}

__global__ void k_zero(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 0.f;
    }
}

__global__ void k_adamw(float *p, const float *g, float *m, float *v, int n,
                        float lr, float wd, float b1, float b2, float eps,
                        float b1t, float b2t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mh = mi / (1.f - b1t);
    float vh = vi / (1.f - b2t);
    p[i] = p[i] - lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}

__global__ void k_sqsum(const float *g, int n, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float loc = (i < n) ? g[i] * g[i] : 0.f;
    __shared__ float sh[256];
    sh[threadIdx.x] = loc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, sh[0]);
    }
}

__global__ void k_scale(float *g, int n, float s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        g[i] *= s;
    }
}

__global__ void k_copy_v(float *V, const float *qkv, int B, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) {
        return;
    }
    int row = i / H;
    int col = i % H;
    V[i] = qkv[row * 3 * H + 2 * H + col];
}

__global__ void k_scatter_v(float *qkv, const float *dV, int B, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) {
        return;
    }
    int row = i / H;
    int col = i % H;
    qkv[row * 3 * H + 2 * H + col] = dV[i];
}

/* PyTorch MHA dropout on attn weights; seq_len=1 → per-head scale of V */
__global__ void k_head_drop(float *V, uint8_t *mask, int B, int nhead, int hd,
                            float drop_p, uint32_t seed, int train)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nmask = B * nhead;
    if (idx >= nmask) {
        return;
    }
    uint8_t keep = 1;
    float scale = 1.f;
    if (train && drop_p > 0.f) {
        uint32_t h = (uint32_t)idx * 747796405u + seed;
        h ^= h >> 17;
        h *= 2246822519u;
        float u = (h >> 8) * (1.f / 16777216.f);
        keep = u >= drop_p;
        scale = keep ? 1.f / (1.f - drop_p) : 0.f;
    }
    mask[idx] = keep;
    float *row = V + (idx / nhead) * (nhead * hd) + (idx % nhead) * hd;
    for (int i = 0; i < hd; i++) {
        row[i] *= scale;
    }
}

__global__ void k_head_drop_bwd(float *dV, const uint8_t *mask, int B, int nhead,
                                int hd, float drop_p, int train)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int H = nhead * hd;
    if (i >= B * H) {
        return;
    }
    if (!train || drop_p <= 0.f) {
        return;
    }
    int row = i / H;
    int col = i % H;
    int head = col / hd;
    dV[i] = mask[row * nhead + head] ? dV[i] / (1.f - drop_p) : 0.f;
}

static int linear_fwd(cublasHandle_t h, const float *X, const float *W,
                      const float *b, float *Y, int B, int out, int in)
{
    if (gemm_xt(h, B, out, in, X, W, Y) != ANN_OK) {
        return ANN_ERR;
    }
    k_add_bias<<<blocks(B * out), 256>>>(Y, b, B, out);
    return ANN_OK;
}

static int linear_bwd(cublasHandle_t h, const float *X, const float *W,
                      const float *dY, float *dX, float *dW, float *db, int B,
                      int out, int in, int acc_dx)
{
    if (dX) {
        if (gemm_dx(h, B, out, in, dY, W, dX, acc_dx ? 1.f : 0.f) != ANN_OK) {
            return ANN_ERR;
        }
    }
    if (gemm_dw(h, B, out, in, dY, X, dW) != ANN_OK) {
        return ANN_ERR;
    }
    k_bias_grad<<<blocks(out), 256>>>(dY, db, B, out);
    return ANN_OK;
}

typedef struct {
    float *qkv[ANN_NLAYERS];
    float *V[ANN_NLAYERS];
    float *attn[ANN_NLAYERS];
    float *res1[ANN_NLAYERS];
    float *ln1x[ANN_NLAYERS];
    float *ln1y[ANN_NLAYERS];
    float *ff_pre[ANN_NLAYERS];
    float *relu[ANN_NLAYERS];
    float *ff_dropin[ANN_NLAYERS];
    float *ff_out[ANN_NLAYERS];
    float *res2[ANN_NLAYERS];
    float *ln2x[ANN_NLAYERS];
    float *mean1[ANN_NLAYERS], *rstd1[ANN_NLAYERS];
    float *mean2[ANN_NLAYERS], *rstd2[ANN_NLAYERS];
    uint8_t *m_attn[ANN_NLAYERS], *m_ffmid[ANN_NLAYERS], *m_ff[ANN_NLAYERS];
    uint8_t *m_aw[ANN_NLAYERS];
    float *h[ANN_NLAYERS + 1];
    float *pred;
    float *ws_ff;
    float *d_h[ANN_NLAYERS + 1];
    float *d_scratch;
} Work;

static Work Wk;
static int wk_ok = 0;
static float *g_dx, *g_dy, *g_dpred, *g_demb, *g_dloss;
static int io_ok = 0;

static void invalidate_static_ws(void)
{
    memset(&Wk, 0, sizeof(Wk));
    wk_ok = 0;
    io_ok = 0;
    g_dx = g_dy = g_dpred = g_demb = g_dloss = NULL;
}

static int work_alloc(int B, int H, int FF)
{
    if (wk_ok) {
        return ANN_OK;
    }
    memset(&Wk, 0, sizeof(Wk));
    Wk.ws_ff = dalloc((size_t)B * FF);
    Wk.pred = dalloc((size_t)B * ANN_OUTSIZE);
    Wk.d_scratch = dalloc((size_t)B * H);
    for (int i = 0; i <= ANN_NLAYERS; i++) {
        Wk.d_h[i] = dalloc((size_t)B * H);
        Wk.h[i] = dalloc((size_t)B * H);
        if (!Wk.d_h[i] || !Wk.h[i]) {
            return ANN_ERR;
        }
    }
    for (int L = 0; L < ANN_NLAYERS; L++) {
        Wk.qkv[L] = dalloc((size_t)B * 3 * H);
        Wk.V[L] = dalloc((size_t)B * H);
        Wk.attn[L] = dalloc((size_t)B * H);
        Wk.res1[L] = dalloc((size_t)B * H);
        Wk.ln1x[L] = dalloc((size_t)B * H);
        Wk.ln1y[L] = dalloc((size_t)B * H);
        Wk.ff_pre[L] = dalloc((size_t)B * FF);
        Wk.relu[L] = dalloc((size_t)B * FF);
        Wk.ff_dropin[L] = dalloc((size_t)B * FF);
        Wk.ff_out[L] = dalloc((size_t)B * H);
        Wk.res2[L] = dalloc((size_t)B * H);
        Wk.ln2x[L] = dalloc((size_t)B * H);
        Wk.mean1[L] = dalloc((size_t)B);
        Wk.rstd1[L] = dalloc((size_t)B);
        Wk.mean2[L] = dalloc((size_t)B);
        Wk.rstd2[L] = dalloc((size_t)B);
        if (!Wk.qkv[L] || !Wk.ln2x[L]) {
            return ANN_ERR;
        }
        cudaMalloc((void **)&Wk.m_attn[L], (size_t)B * H);
        cudaMalloc((void **)&Wk.m_ff[L], (size_t)B * H);
        cudaMalloc((void **)&Wk.m_ffmid[L], (size_t)B * FF);
        cudaMalloc((void **)&Wk.m_aw[L], (size_t)B * ANN_NHEAD);
        if (!Wk.m_attn[L] || !Wk.m_ff[L] || !Wk.m_ffmid[L] || !Wk.m_aw[L]) {
            return ANN_ERR;
        }
    }
    wk_ok = 1;
    return ANN_OK;
}

static int io_alloc(int B, int I, int H, int O)
{
    if (io_ok) {
        return ANN_OK;
    }
    g_dx = dalloc((size_t)B * I);
    g_dy = dalloc((size_t)B * O);
    g_dpred = dalloc((size_t)B * O);
    g_demb = dalloc((size_t)B * H);
    g_dloss = dalloc(1);
    if (!g_dx || !g_dy || !g_dpred || !g_demb || !g_dloss) {
        return ANN_ERR;
    }
    io_ok = 1;
    return ANN_OK;
}

static int alloc_layer(EncLayer *L, int H, int FF)
{
    memset(L, 0, sizeof(*L));
#define A(p, n)                                                                \
    do {                                                                       \
        p = dalloc((size_t)(n));                                               \
        if (!(p) && (n) > 0)                                                   \
            return ANN_ERR;                                                    \
    } while (0)
    A(L->W_qkv, 3 * H * H);
    A(L->b_qkv, 3 * H);
    A(L->W_out, H * H);
    A(L->b_out, H);
    A(L->W_l1, FF * H);
    A(L->b_l1, FF);
    A(L->W_l2, H * FF);
    A(L->b_l2, H);
    A(L->n1w, H);
    A(L->n1b, H);
    A(L->n2w, H);
    A(L->n2b, H);
    A(L->gW_qkv, 3 * H * H);
    A(L->gb_qkv, 3 * H);
    A(L->gW_out, H * H);
    A(L->gb_out, H);
    A(L->gW_l1, FF * H);
    A(L->gb_l1, FF);
    A(L->gW_l2, H * FF);
    A(L->gb_l2, H);
    A(L->gn1w, H);
    A(L->gn1b, H);
    A(L->gn2w, H);
    A(L->gn2b, H);
    A(L->mW_qkv, 3 * H * H);
    A(L->vW_qkv, 3 * H * H);
    A(L->mb_qkv, 3 * H);
    A(L->vb_qkv, 3 * H);
    A(L->mW_out, H * H);
    A(L->vW_out, H * H);
    A(L->mb_out, H);
    A(L->vb_out, H);
    A(L->mW_l1, FF * H);
    A(L->vW_l1, FF * H);
    A(L->mb_l1, FF);
    A(L->vb_l1, FF);
    A(L->mW_l2, H * FF);
    A(L->vW_l2, H * FF);
    A(L->mb_l2, H);
    A(L->vb_l2, H);
    A(L->mn1w, H);
    A(L->vn1w, H);
    A(L->mn1b, H);
    A(L->vn1b, H);
    A(L->mn2w, H);
    A(L->vn2w, H);
    A(L->mn2b, H);
    A(L->vn2b, H);
#undef A
    return ANN_OK;
}

static void free_layer(EncLayer *L)
{
#define F(p) dfree(p)
    F(L->W_qkv);
    F(L->b_qkv);
    F(L->W_out);
    F(L->b_out);
    F(L->W_l1);
    F(L->b_l1);
    F(L->W_l2);
    F(L->b_l2);
    F(L->n1w);
    F(L->n1b);
    F(L->n2w);
    F(L->n2b);
    F(L->gW_qkv);
    F(L->gb_qkv);
    F(L->gW_out);
    F(L->gb_out);
    F(L->gW_l1);
    F(L->gb_l1);
    F(L->gW_l2);
    F(L->gb_l2);
    F(L->gn1w);
    F(L->gn1b);
    F(L->gn2w);
    F(L->gn2b);
    F(L->mW_qkv);
    F(L->vW_qkv);
    F(L->mb_qkv);
    F(L->vb_qkv);
    F(L->mW_out);
    F(L->vW_out);
    F(L->mb_out);
    F(L->vb_out);
    F(L->mW_l1);
    F(L->vW_l1);
    F(L->mb_l1);
    F(L->vb_l1);
    F(L->mW_l2);
    F(L->vW_l2);
    F(L->mb_l2);
    F(L->vb_l2);
    F(L->mn1w);
    F(L->vn1w);
    F(L->mn1b);
    F(L->vn1b);
    F(L->mn2w);
    F(L->vn2w);
    F(L->mn2b);
    F(L->vn2b);
#undef F
}

static void host_xavier(float *h, int out, int in, uint32_t *rng)
{
    float bound = sqrtf(6.f / (float)(in + out));
    int n = out * in;
    for (int i = 0; i < n; i++) {
        *rng = *rng * 1664525u + 1013904223u;
        float u = ((*rng) >> 8) * (1.f / 16777216.f);
        h[i] = u * 2.f * bound - bound;
    }
}

static int copy_xavier(float *d, int out, int in, uint32_t *rng)
{
    int n = out * in;
    float *h = (float *)malloc((size_t)n * sizeof(float));
    if (!h) {
        return ANN_ERR;
    }
    host_xavier(h, out, in, rng);
    cudaMemcpy(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return ANN_OK;
}

static int fill_ones(float *d, int n)
{
    float *h = (float *)malloc((size_t)n * sizeof(float));
    if (!h) {
        return ANN_ERR;
    }
    for (int i = 0; i < n; i++) {
        h[i] = 1.f;
    }
    cudaMemcpy(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return ANN_OK;
}

int model_create(Model *m, const char *code, int dropout_enable)
{
    memset(m, 0, sizeof(*m));
    snprintf(m->code, sizeof(m->code), "%s", code);
    m->insize = ANN_INSIZE;
    m->hidden = ANN_HIDDEN;
    m->outsize = ANN_OUTSIZE;
    m->nhead = ANN_NHEAD;
    m->nlayers = ANN_NLAYERS;
    m->d_ff = ANN_D_FF;
    m->dropout = dropout_enable ? ANN_DROPOUT : 0.f;
    if (!cuda_runtime_ok()) {
        fprintf(stderr, "no CUDA device\n");
        return ANN_ERR;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed %s: %s\n", code,
                cudaGetErrorString(cudaGetLastError()));
        return ANN_ERR;
    }
    cublasHandle_t h = NULL;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed %s status=%d cuda=%s\n", code,
                (int)st, cudaGetErrorString(cudaGetLastError()));
        return ANN_ERR;
    }
    m->cublas = h;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    m->W_emb = dalloc((size_t)H * I);
    m->b_emb = dalloc((size_t)H);
    m->W_fc = dalloc((size_t)O * H);
    m->b_fc = dalloc((size_t)O);
    m->gW_emb = dalloc((size_t)H * I);
    m->gb_emb = dalloc((size_t)H);
    m->gW_fc = dalloc((size_t)O * H);
    m->gb_fc = dalloc((size_t)O);
    m->mW_emb = dalloc((size_t)H * I);
    m->vW_emb = dalloc((size_t)H * I);
    m->mb_emb = dalloc((size_t)H);
    m->vb_emb = dalloc((size_t)H);
    m->mW_fc = dalloc((size_t)O * H);
    m->vW_fc = dalloc((size_t)O * H);
    m->mb_fc = dalloc((size_t)O);
    m->vb_fc = dalloc((size_t)O);
    m->pe0 = dalloc((size_t)H);
    if (!m->W_emb || !m->W_fc || !m->pe0) {
        fprintf(stderr, "cudaMalloc failed %s: %s\n", code,
                cudaGetErrorString(cudaGetLastError()));
        model_destroy(m);
        return ANN_ERR;
    }
    for (int i = 0; i < ANN_NLAYERS; i++) {
        if (alloc_layer(&m->layers[i], H, FF) != ANN_OK) {
            fprintf(stderr, "alloc_layer failed %s layer %d: %s\n", code, i,
                    cudaGetErrorString(cudaGetLastError()));
            model_destroy(m);
            return ANN_ERR;
        }
    }
    if (io_alloc(ANN_MAX_BATCH, I, H, O) != ANN_OK ||
        work_alloc(ANN_MAX_BATCH, H, FF) != ANN_OK) {
        fprintf(stderr, "workspace alloc failed %s: %s\n", code,
                cudaGetErrorString(cudaGetLastError()));
        model_destroy(m);
        return ANN_ERR;
    }
    float *pe = (float *)malloc((size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) {
        pe[i] = (i % 2 == 0) ? 0.f : 1.f;
    }
    cudaMemcpy(m->pe0, pe, (size_t)H * sizeof(float), cudaMemcpyHostToDevice);
    free(pe);
    model_xavier_init(m, ANN_SEED);
    return ANN_OK;
}

void model_destroy(Model *m)
{
    if (!m) {
        return;
    }
    if (m->cublas) {
        cublasDestroy((cublasHandle_t)m->cublas);
    }
    dfree(m->W_emb);
    dfree(m->b_emb);
    dfree(m->W_fc);
    dfree(m->b_fc);
    dfree(m->gW_emb);
    dfree(m->gb_emb);
    dfree(m->gW_fc);
    dfree(m->gb_fc);
    dfree(m->mW_emb);
    dfree(m->vW_emb);
    dfree(m->mb_emb);
    dfree(m->vb_emb);
    dfree(m->mW_fc);
    dfree(m->vW_fc);
    dfree(m->mb_fc);
    dfree(m->vb_fc);
    dfree(m->pe0);
    for (int i = 0; i < ANN_NLAYERS; i++) {
        free_layer(&m->layers[i]);
    }
    memset(m, 0, sizeof(*m));
}

void model_xavier_init(Model *m, uint32_t seed)
{
    uint32_t rng = seed ? seed : 1u;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    copy_xavier(m->W_emb, H, I, &rng);
    copy_xavier(m->W_fc, O, H, &rng);
    cudaMemset(m->b_emb, 0, (size_t)H * sizeof(float));
    cudaMemset(m->b_fc, 0, (size_t)O * sizeof(float));
    for (int i = 0; i < m->nlayers; i++) {
        EncLayer *L = &m->layers[i];
        copy_xavier(L->W_qkv, 3 * H, H, &rng);
        copy_xavier(L->W_out, H, H, &rng);
        copy_xavier(L->W_l1, FF, H, &rng);
        copy_xavier(L->W_l2, H, FF, &rng);
        cudaMemset(L->b_qkv, 0, (size_t)3 * H * sizeof(float));
        cudaMemset(L->b_out, 0, (size_t)H * sizeof(float));
        cudaMemset(L->b_l1, 0, (size_t)FF * sizeof(float));
        cudaMemset(L->b_l2, 0, (size_t)H * sizeof(float));
        fill_ones(L->n1w, H);
        fill_ones(L->n2w, H);
        cudaMemset(L->n1b, 0, (size_t)H * sizeof(float));
        cudaMemset(L->n2b, 0, (size_t)H * sizeof(float));
    }
}

void model_zero_grad(Model *m)
{
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    cudaMemset(m->gW_emb, 0, (size_t)H * I * sizeof(float));
    cudaMemset(m->gb_emb, 0, (size_t)H * sizeof(float));
    cudaMemset(m->gW_fc, 0, (size_t)O * H * sizeof(float));
    cudaMemset(m->gb_fc, 0, (size_t)O * sizeof(float));
    for (int i = 0; i < m->nlayers; i++) {
        EncLayer *L = &m->layers[i];
        cudaMemset(L->gW_qkv, 0, (size_t)3 * H * H * sizeof(float));
        cudaMemset(L->gb_qkv, 0, (size_t)3 * H * sizeof(float));
        cudaMemset(L->gW_out, 0, (size_t)H * H * sizeof(float));
        cudaMemset(L->gb_out, 0, (size_t)H * sizeof(float));
        cudaMemset(L->gW_l1, 0, (size_t)FF * H * sizeof(float));
        cudaMemset(L->gb_l1, 0, (size_t)FF * sizeof(float));
        cudaMemset(L->gW_l2, 0, (size_t)H * FF * sizeof(float));
        cudaMemset(L->gb_l2, 0, (size_t)H * sizeof(float));
        cudaMemset(L->gn1w, 0, (size_t)H * sizeof(float));
        cudaMemset(L->gn1b, 0, (size_t)H * sizeof(float));
        cudaMemset(L->gn2w, 0, (size_t)H * sizeof(float));
        cudaMemset(L->gn2b, 0, (size_t)H * sizeof(float));
    }
}

static int enc_fwd(cublasHandle_t h, EncLayer *L, const float *src, float *dst,
                   int li, int B, int H, int FF, float drop_p, int train,
                   uint32_t seed)
{
    if (linear_fwd(h, src, L->W_qkv, L->b_qkv, Wk.qkv[li], B, 3 * H, H) != ANN_OK) {
        return ANN_ERR;
    }
    k_copy_v<<<blocks(B * H), 256>>>(Wk.V[li], Wk.qkv[li], B, H);
    k_head_drop<<<blocks(B * ANN_NHEAD), 256>>>(Wk.V[li], Wk.m_aw[li], B, ANN_NHEAD,
                                                ANN_HEAD_DIM, drop_p, seed + 3u,
                                                train);
    if (linear_fwd(h, Wk.V[li], L->W_out, L->b_out, Wk.attn[li], B, H, H) != ANN_OK) {
        return ANN_ERR;
    }
    k_dropout<<<blocks(B * H), 256>>>(Wk.res1[li], Wk.m_attn[li], Wk.attn[li], B * H,
                                      drop_p, seed + 17u, train);
    k_add<<<blocks(B * H), 256>>>(Wk.ln1x[li], src, Wk.res1[li], B * H);
    k_ln_fwd<<<B, 256>>>(Wk.ln1x[li], Wk.ln1y[li], L->n1w, L->n1b, Wk.mean1[li],
                         Wk.rstd1[li], B, H, ANN_LN_EPS);
    if (linear_fwd(h, Wk.ln1y[li], L->W_l1, L->b_l1, Wk.ff_pre[li], B, FF, H) !=
        ANN_OK) {
        return ANN_ERR;
    }
    k_relu<<<blocks(B * FF), 256>>>(Wk.relu[li], Wk.ff_pre[li], B * FF);
    k_dropout<<<blocks(B * FF), 256>>>(Wk.ff_dropin[li], Wk.m_ffmid[li], Wk.relu[li],
                                       B * FF, drop_p, seed + 1000u, train);
    if (linear_fwd(h, Wk.ff_dropin[li], L->W_l2, L->b_l2, Wk.ff_out[li], B, H,
                   FF) != ANN_OK) {
        return ANN_ERR;
    }
    k_dropout<<<blocks(B * H), 256>>>(Wk.res2[li], Wk.m_ff[li], Wk.ff_out[li], B * H,
                                      drop_p, seed + 2000u, train);
    k_add<<<blocks(B * H), 256>>>(Wk.ln2x[li], Wk.ln1y[li], Wk.res2[li], B * H);
    k_ln_fwd<<<B, 256>>>(Wk.ln2x[li], dst, L->n2w, L->n2b, Wk.mean2[li],
                         Wk.rstd2[li], B, H, ANN_LN_EPS);
    return ANN_OK;
}

static int enc_bwd(cublasHandle_t h, EncLayer *L, const float *src,
                   const float *d_dst, float *d_src, int li, int B, int H, int FF,
                   float drop_p, int train)
{
    float *d_ln2x = Wk.attn[li];
    k_ln_bwd<<<B, 256>>>(Wk.ln2x[li], d_dst, L->n2w, Wk.mean2[li], Wk.rstd2[li],
                         d_ln2x, L->gn2w, L->gn2b, B, H);

    float *d_ffdrop = Wk.ff_out[li];
    k_dropout_bwd<<<blocks(B * H), 256>>>(d_ffdrop, Wk.m_ff[li], d_ln2x, B * H,
                                          drop_p, train);
    if (linear_bwd(h, Wk.ff_dropin[li], L->W_l2, d_ffdrop, Wk.ws_ff, L->gW_l2,
                   L->gb_l2, B, H, FF, 0) != ANN_OK) {
        return ANN_ERR;
    }
    k_dropout_bwd<<<blocks(B * FF), 256>>>(Wk.ws_ff, Wk.m_ffmid[li], Wk.ws_ff, B * FF,
                                           drop_p, train);
    k_relu_bwd<<<blocks(B * FF), 256>>>(Wk.ws_ff, Wk.relu[li], Wk.ws_ff, B * FF);

    float *d_src1n = Wk.res1[li];
    if (linear_bwd(h, Wk.ln1y[li], L->W_l1, Wk.ws_ff, d_src1n, L->gW_l1, L->gb_l1,
                   B, FF, H, 0) != ANN_OK) {
        return ANN_ERR;
    }
    k_axpy<<<blocks(B * H), 256>>>(d_src1n, d_ln2x, B * H);

    float *d_ln1x = Wk.ln1y[li];
    k_ln_bwd<<<B, 256>>>(Wk.ln1x[li], d_src1n, L->n1w, Wk.mean1[li], Wk.rstd1[li],
                         d_ln1x, L->gn1w, L->gn1b, B, H);
    cudaMemcpy(d_src, d_ln1x, (size_t)B * H * sizeof(float),
               cudaMemcpyDeviceToDevice);

    float *d_attn = Wk.d_scratch;
    k_dropout_bwd<<<blocks(B * H), 256>>>(d_attn, Wk.m_attn[li], d_ln1x, B * H,
                                          drop_p, train);
    float *d_V = Wk.res2[li];
    if (linear_bwd(h, Wk.V[li], L->W_out, d_attn, d_V, L->gW_out, L->gb_out, B, H,
                   H, 0) != ANN_OK) {
        return ANN_ERR;
    }
    k_head_drop_bwd<<<blocks(B * H), 256>>>(d_V, Wk.m_aw[li], B, ANN_NHEAD,
                                            ANN_HEAD_DIM, drop_p, train);
    k_zero<<<blocks(B * 3 * H), 256>>>(Wk.qkv[li], B * 3 * H);
    k_scatter_v<<<blocks(B * H), 256>>>(Wk.qkv[li], d_V, B, H);
    if (linear_bwd(h, src, L->W_qkv, Wk.qkv[li], d_src, L->gW_qkv, L->gb_qkv, B,
                   3 * H, H, 1) != ANN_OK) {
        return ANN_ERR;
    }
    return ANN_OK;
}

typedef struct {
    float *p, *g, *m, *v;
    int n;
} PSlot;

static int collect_params(Model *m, PSlot *s)
{
    int n = 0, H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
#define ADD(P, G, M, V, N)                                                     \
    do {                                                                       \
        s[n].p = (P);                                                          \
        s[n].g = (G);                                                          \
        s[n].m = (M);                                                          \
        s[n].v = (V);                                                          \
        s[n].n = (N);                                                          \
        n++;                                                                   \
    } while (0)
    ADD(m->W_emb, m->gW_emb, m->mW_emb, m->vW_emb, H * I);
    ADD(m->b_emb, m->gb_emb, m->mb_emb, m->vb_emb, H);
    ADD(m->W_fc, m->gW_fc, m->mW_fc, m->vW_fc, O * H);
    ADD(m->b_fc, m->gb_fc, m->mb_fc, m->vb_fc, O);
    for (int i = 0; i < m->nlayers; i++) {
        EncLayer *L = &m->layers[i];
        ADD(L->W_qkv, L->gW_qkv, L->mW_qkv, L->vW_qkv, 3 * H * H);
        ADD(L->b_qkv, L->gb_qkv, L->mb_qkv, L->vb_qkv, 3 * H);
        ADD(L->W_out, L->gW_out, L->mW_out, L->vW_out, H * H);
        ADD(L->b_out, L->gb_out, L->mb_out, L->vb_out, H);
        ADD(L->W_l1, L->gW_l1, L->mW_l1, L->vW_l1, FF * H);
        ADD(L->b_l1, L->gb_l1, L->mb_l1, L->vb_l1, FF);
        ADD(L->W_l2, L->gW_l2, L->mW_l2, L->vW_l2, H * FF);
        ADD(L->b_l2, L->gb_l2, L->mb_l2, L->vb_l2, H);
        ADD(L->n1w, L->gn1w, L->mn1w, L->vn1w, H);
        ADD(L->n1b, L->gn1b, L->mn1b, L->vn1b, H);
        ADD(L->n2w, L->gn2w, L->mn2w, L->vn2w, H);
        ADD(L->n2b, L->gn2b, L->mn2b, L->vn2b, H);
    }
#undef ADD
    return n;
}

static int net_forward(Model *m, int B, uint32_t seed)
{
    cublasHandle_t h = (cublasHandle_t)m->cublas;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    int train = m->training;
    float drop_p = m->dropout;
    if (linear_fwd(h, g_dx, m->W_emb, m->b_emb, Wk.h[0], B, H, I) != ANN_OK) {
        return ANN_ERR;
    }
    k_add_pe<<<blocks(B * H), 256>>>(Wk.h[0], Wk.h[0], m->pe0, B, H);
    for (int li = 0; li < m->nlayers; li++) {
        if (enc_fwd(h, &m->layers[li], Wk.h[li], Wk.h[li + 1], li, B, H, FF,
                    drop_p, train, seed + (uint32_t)li * 4096u) != ANN_OK) {
            return ANN_ERR;
        }
    }
    if (linear_fwd(h, Wk.h[m->nlayers], m->W_fc, m->b_fc, Wk.pred, B, O, H) !=
        ANN_OK) {
        return ANN_ERR;
    }
    return ANN_OK;
}

int model_forward(Model *m, const float *x_host, int batch, float *y_host)
{
    if (batch <= 0 || batch > ANN_MAX_BATCH) {
        return ANN_ERR;
    }
    int saved = m->training;
    m->training = 0;
    CUDA_CHECK(cudaMemcpy(g_dx, x_host, (size_t)batch * m->insize * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (net_forward(m, batch, 0) != ANN_OK) {
        m->training = saved;
        return ANN_ERR;
    }
    CUDA_CHECK(cudaMemcpy(y_host, Wk.pred,
                          (size_t)batch * m->outsize * sizeof(float),
                          cudaMemcpyDeviceToHost));
    m->training = saved;
    return ANN_OK;
}

int model_forward_bwd(Model *m, const float *x_host, const float *y_host,
                      int batch, float *loss_out)
{
    if (batch <= 0 || batch > ANN_MAX_BATCH) {
        return ANN_ERR;
    }
    cublasHandle_t h = (cublasHandle_t)m->cublas;
    int H = m->hidden, I = m->insize, O = m->outsize;
    int n = batch * O;
    CUDA_CHECK(cudaMemcpy(g_dx, x_host, (size_t)batch * I * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dy, y_host, (size_t)batch * O * sizeof(float),
                          cudaMemcpyHostToDevice));
    uint32_t seed = (uint32_t)m->adam_step * 10007u + 1u;
    if (net_forward(m, batch, seed) != ANN_OK) {
        return ANN_ERR;
    }
    k_zero<<<1, 32>>>(g_dloss, 1);
    k_l1_bwd<<<blocks(n), 256>>>(Wk.pred, g_dy, g_dpred, n, g_dloss);
    float loss_sum = 0.f;
    CUDA_CHECK(cudaMemcpy(&loss_sum, g_dloss, sizeof(float), cudaMemcpyDeviceToHost));
    if (loss_out) {
        *loss_out = loss_sum / (float)n;
    }
    if (linear_bwd(h, Wk.h[m->nlayers], m->W_fc, g_dpred, Wk.d_h[m->nlayers],
                   m->gW_fc, m->gb_fc, batch, O, H, 0) != ANN_OK) {
        return ANN_ERR;
    }
    for (int li = m->nlayers - 1; li >= 0; li--) {
        if (enc_bwd(h, &m->layers[li], Wk.h[li], Wk.d_h[li + 1], Wk.d_h[li], li,
                    batch, H, m->d_ff, m->dropout, m->training) != ANN_OK) {
            return ANN_ERR;
        }
    }
    if (linear_bwd(h, g_dx, m->W_emb, Wk.d_h[0], NULL, m->gW_emb, m->gb_emb,
                   batch, H, I, 0) != ANN_OK) {
        return ANN_ERR;
    }
    return ANN_OK;
}

int model_adamw_step(Model *m, float lr, float weight_decay)
{
    PSlot slots[64];
    int n = collect_params(m, slots);
    m->adam_step += 1;
    float b1t = powf(ANN_ADAM_B1, (float)m->adam_step);
    float b2t = powf(ANN_ADAM_B2, (float)m->adam_step);
    for (int i = 0; i < n; i++) {
        k_adamw<<<blocks(slots[i].n), 256>>>(
            slots[i].p, slots[i].g, slots[i].m, slots[i].v, slots[i].n, lr,
            weight_decay, ANN_ADAM_B1, ANN_ADAM_B2, ANN_ADAM_EPS, b1t, b2t);
    }
    return ANN_OK;
}

int model_clip_grad(Model *m, float max_norm)
{
    PSlot slots[64];
    int n = collect_params(m, slots);
    k_zero<<<1, 32>>>(g_dloss, 1);
    for (int i = 0; i < n; i++) {
        k_sqsum<<<blocks(slots[i].n), 256>>>(slots[i].g, slots[i].n, g_dloss);
    }
    float ss = 0.f;
    CUDA_CHECK(cudaMemcpy(&ss, g_dloss, sizeof(float), cudaMemcpyDeviceToHost));
    float total = sqrtf(ss);
    if (total > max_norm) {
        float s = max_norm / (total + 1e-6f);
        for (int i = 0; i < n; i++) {
            k_scale<<<blocks(slots[i].n), 256>>>(slots[i].g, slots[i].n, s);
        }
    }
    return ANN_OK;
}

float model_cosine_lr(float base_lr, int ep, int tmax)
{
    if (ep <= 1 || tmax <= 0) {
        return base_lr;
    }
    float t = (float)(ep - 1) / (float)tmax;
    return 0.5f * base_lr * (1.f + cosf(3.14159265358979323846f * t));
}

static void gather_batch(const Dataset *ds, const int *idx, int start, int B,
                         float *xb, float *yb)
{
    int I = ANN_INSIZE, O = ANN_OUTSIZE;
    for (int i = 0; i < B; i++) {
        int j = idx[start + i];
        memcpy(xb + (size_t)i * I, ds->x + (size_t)j * I, (size_t)I * sizeof(float));
        memcpy(yb + (size_t)i * O, ds->y + (size_t)j * O, (size_t)O * sizeof(float));
    }
}

int model_train(Model *m, const Dataset *ds, int epoch, float lr,
                float loss_threshold, int shuffle)
{
    if (!ds || !ds->has_y || ds->n <= 0 || epoch <= 0) {
        return ANN_ERR;
    }
    m->interrupt = 0;
    m->training = 1;
    int *idx = (int *)malloc((size_t)ds->n * sizeof(int));
    float *xb = (float *)malloc((size_t)ANN_MAX_BATCH * ANN_INSIZE * sizeof(float));
    float *yb = (float *)malloc((size_t)ANN_MAX_BATCH * ANN_OUTSIZE * sizeof(float));
    float *hist = (float *)malloc(100 * sizeof(float));
    if (!idx || !xb || !yb || !hist) {
        free(idx);
        free(xb);
        free(yb);
        free(hist);
        return ANN_ERR;
    }
    uint32_t rng = ANN_SEED;
    int nhist = 0;
    float loss_one = 0.f;
    char path[256];
    weights_bin_path(m->code, path, sizeof(path));

    for (int ep = 0; ep < epoch; ep++) {
        if (m->interrupt) {
            break;
        }
        if (shuffle) {
            dataset_shuffle_idx(idx, ds->n, &rng);
        } else {
            for (int i = 0; i < ds->n; i++) {
                idx[i] = i;
            }
        }
        loss_one = 0.f;
        float lr_now = model_cosine_lr(lr, ep, epoch);
        for (int start = 0; start < ds->n; start += ANN_MAX_BATCH) {
            if (m->interrupt) {
                break;
            }
            int B = ds->n - start;
            if (B > ANN_MAX_BATCH) {
                B = ANN_MAX_BATCH;
            }
            gather_batch(ds, idx, start, B, xb, yb);
            model_zero_grad(m);
            float batch_loss = 0.f;
            if (model_forward_bwd(m, xb, yb, B, &batch_loss) != ANN_OK) {
                free(idx);
                free(xb);
                free(yb);
                free(hist);
                m->training = 0;
                return ANN_ERR;
            }
            if (model_clip_grad(m, ANN_CLIP) != ANN_OK ||
                model_adamw_step(m, lr_now, ANN_WD) != ANN_OK) {
                free(idx);
                free(xb);
                free(yb);
                free(hist);
                m->training = 0;
                return ANN_ERR;
            }
            loss_one += batch_loss;
        }
        if (nhist < 100) {
            hist[nhist++] = loss_one;
        } else {
            memmove(hist, hist + 1, 99 * sizeof(float));
            hist[99] = loss_one;
        }
        if (m->interrupt) {
            printf("%s [%d/%d], loss: %.6f\n", m->code, ep + 1, epoch, loss_one);
            break;
        }
        if (ep % 5000 == 0) {
            float avg = 0.f;
            for (int i = 0; i < nhist; i++) {
                avg += hist[i];
            }
            avg /= (float)nhist;
            time_t now = time(NULL);
            struct tm *tmv = localtime(&now);
            printf("%s [%d/%d] %02d:%02d:%02d | %.6f\n", m->code, ep + 1, epoch,
                   tmv ? tmv->tm_hour : 0, tmv ? tmv->tm_min : 0,
                   tmv ? tmv->tm_sec : 0, avg);
            if (ep > 100 && loss_one > 2.f * avg) {
                printf("Warning: Loss increased significantly! Current: %.6f, "
                       "Average: %.6f\n",
                       loss_one, avg);
            }
            if (ep > 100) {
                udp_log_loss(m->code, avg);
            }
        }
        if (ep > 1000 && ep % 50000 == 1) {
            weights_save(m, path);
            printf("save model %s\n", m->code);
        }
        if (loss_one < loss_threshold && loss_one > 0.f && ep > 1000) {
            printf("%s [%d/%d],loss:%.6f\n", m->code, ep + 1, epoch, loss_one);
            printf("%s: %f < %f \n", m->code, loss_one, loss_threshold);
            m->interrupt = 0;
            m->training = 0;
            free(idx);
            free(xb);
            free(yb);
            free(hist);
            return ANN_OK;
        }
    }
    m->training = 0;
    free(idx);
    free(xb);
    free(yb);
    free(hist);
    return ANN_OK;
}

int model_predict(Model *m, const Dataset *ds, float *pred_out)
{
    if (!ds || ds->n <= 0 || !pred_out) {
        return ANN_ERR;
    }
    int saved = m->training;
    m->training = 0;
    for (int start = 0; start < ds->n; start += ANN_MAX_BATCH) {
        int B = ds->n - start;
        if (B > ANN_MAX_BATCH) {
            B = ANN_MAX_BATCH;
        }
        if (model_forward(m, ds->x + (size_t)start * ANN_INSIZE, B,
                          pred_out + (size_t)start * ANN_OUTSIZE) != ANN_OK) {
            m->training = saved;
            return ANN_ERR;
        }
    }
    m->training = saved;
    return ANN_OK;
}
