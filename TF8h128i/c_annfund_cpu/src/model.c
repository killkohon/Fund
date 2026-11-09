#include "model.h"
#include "weights.h"

#ifdef ANN_BLAS_BUILTIN
#include "cblas_builtin.h"
#else
#include <cblas.h>
/* OpenBLAS pulls complex.h which defines macro I (imaginary unit). */
#ifdef I
#undef I
#endif
#endif
#include <arpa/inet.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define ANN_UDP_HOST "8.138.255.148"
#define ANN_UDP_PORT 48888

/* Verbatim port of model.cu udp_log_loss (POSIX sockets, non-blocking). */
static void udp_log_loss(const char *code, float avg)
{
    static int fd = -2;
    char msg[64];
    int n = snprintf(msg, sizeof(msg), "%s %.6f", code, avg);
    if (n <= 0) {
        return;
    }
    if (fd == -2) {
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd >= 0) {
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0) {
                fcntl(fd, F_SETFL, flags | O_NONBLOCK);
            }
        }
    }
    if (fd < 0) {
        return;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)ANN_UDP_PORT);
    if (inet_pton(AF_INET, ANN_UDP_HOST, &addr.sin_addr) != 1) {
        return;
    }
    (void)sendto(fd, msg, (size_t)n, 0, (struct sockaddr *)&addr, sizeof(addr));
}

/* Per-Model workspace. Opaque to the rest of the project via
 * `struct Work *` in model.h; fully defined here so all allocation
 * lives and dies with the Model that owns it. */
typedef struct Work {
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
    /* IO buffers (formerly static g_dx/g_dy/g_dpred/g_demb in model.cu) */
    float *x_buf, *y_buf, *dpred, *demb;
} Work;

static float *halloc(size_t n)
{
    if (n == 0) return NULL;
    return (float *)calloc(n, sizeof(float));
}

static uint8_t *halloc_u8(size_t n)
{
    if (n == 0) return NULL;
    return (uint8_t *)calloc(n, sizeof(uint8_t));
}

static void hfree(void *p) { free(p); }

static int work_alloc(Work *wk, int B, int H, int FF)
{
    wk->ws_ff = halloc((size_t)B * FF);
    wk->pred = halloc((size_t)B * ANN_OUTSIZE);
    wk->d_scratch = halloc((size_t)B * H);
    for (int i = 0; i <= ANN_NLAYERS; i++) {
        wk->d_h[i] = halloc((size_t)B * H);
        wk->h[i] = halloc((size_t)B * H);
        if (!wk->d_h[i] || !wk->h[i]) return ANN_ERR;
    }
    for (int L = 0; L < ANN_NLAYERS; L++) {
        wk->qkv[L] = halloc((size_t)B * 3 * H);
        wk->V[L] = halloc((size_t)B * H);
        wk->attn[L] = halloc((size_t)B * H);
        wk->res1[L] = halloc((size_t)B * H);
        wk->ln1x[L] = halloc((size_t)B * H);
        wk->ln1y[L] = halloc((size_t)B * H);
        wk->ff_pre[L] = halloc((size_t)B * FF);
        wk->relu[L] = halloc((size_t)B * FF);
        wk->ff_dropin[L] = halloc((size_t)B * FF);
        wk->ff_out[L] = halloc((size_t)B * H);
        wk->res2[L] = halloc((size_t)B * H);
        wk->ln2x[L] = halloc((size_t)B * H);
        wk->mean1[L] = halloc((size_t)B);
        wk->rstd1[L] = halloc((size_t)B);
        wk->mean2[L] = halloc((size_t)B);
        wk->rstd2[L] = halloc((size_t)B);
        wk->m_attn[L] = halloc_u8((size_t)B * H);
        wk->m_ff[L] = halloc_u8((size_t)B * H);
        wk->m_ffmid[L] = halloc_u8((size_t)B * FF);
        wk->m_aw[L] = halloc_u8((size_t)B * ANN_NHEAD);
        if (!wk->qkv[L] || !wk->ln2x[L] || !wk->m_attn[L] || !wk->m_ff[L] ||
            !wk->m_ffmid[L] || !wk->m_aw[L]) {
            return ANN_ERR;
        }
    }
    return ANN_OK;
}

static void work_free(Work *wk)
{
    if (!wk) return;
    hfree(wk->ws_ff);
    hfree(wk->pred);
    hfree(wk->d_scratch);
    for (int i = 0; i <= ANN_NLAYERS; i++) {
        hfree(wk->d_h[i]);
        hfree(wk->h[i]);
    }
    for (int L = 0; L < ANN_NLAYERS; L++) {
        hfree(wk->qkv[L]); hfree(wk->V[L]); hfree(wk->attn[L]);
        hfree(wk->res1[L]); hfree(wk->ln1x[L]); hfree(wk->ln1y[L]);
        hfree(wk->ff_pre[L]); hfree(wk->relu[L]); hfree(wk->ff_dropin[L]);
        hfree(wk->ff_out[L]); hfree(wk->res2[L]); hfree(wk->ln2x[L]);
        hfree(wk->mean1[L]); hfree(wk->rstd1[L]);
        hfree(wk->mean2[L]); hfree(wk->rstd2[L]);
        hfree(wk->m_attn[L]); hfree(wk->m_ff[L]);
        hfree(wk->m_ffmid[L]); hfree(wk->m_aw[L]);
    }
    hfree(wk->x_buf); hfree(wk->y_buf);
    hfree(wk->dpred); hfree(wk->demb);
}

static int io_alloc(Work *wk, int B, int I, int H, int O)
{
    wk->x_buf = halloc((size_t)B * I);
    wk->y_buf = halloc((size_t)B * O);
    wk->dpred = halloc((size_t)B * O);
    wk->demb = halloc((size_t)B * H);
    if (!wk->x_buf || !wk->y_buf || !wk->dpred || !wk->demb) return ANN_ERR;
    return ANN_OK;
}

static int alloc_layer(EncLayer *L, int H, int FF)
{
    memset(L, 0, sizeof(*L));
#define A(p, n)                                                                \
    do {                                                                       \
        p = halloc((size_t)(n));                                               \
        if (!(p) && (n) > 0) return ANN_ERR;                                   \
    } while (0)
    A(L->W_qkv, 3 * H * H); A(L->b_qkv, 3 * H);
    A(L->W_out, H * H); A(L->b_out, H);
    A(L->W_l1, FF * H); A(L->b_l1, FF);
    A(L->W_l2, H * FF); A(L->b_l2, H);
    A(L->n1w, H); A(L->n1b, H); A(L->n2w, H); A(L->n2b, H);
    A(L->gW_qkv, 3 * H * H); A(L->gb_qkv, 3 * H);
    A(L->gW_out, H * H); A(L->gb_out, H);
    A(L->gW_l1, FF * H); A(L->gb_l1, FF);
    A(L->gW_l2, H * FF); A(L->gb_l2, H);
    A(L->gn1w, H); A(L->gn1b, H); A(L->gn2w, H); A(L->gn2b, H);
    A(L->mW_qkv, 3 * H * H); A(L->vW_qkv, 3 * H * H);
    A(L->mb_qkv, 3 * H); A(L->vb_qkv, 3 * H);
    A(L->mW_out, H * H); A(L->vW_out, H * H);
    A(L->mb_out, H); A(L->vb_out, H);
    A(L->mW_l1, FF * H); A(L->vW_l1, FF * H);
    A(L->mb_l1, FF); A(L->vb_l1, FF);
    A(L->mW_l2, H * FF); A(L->vW_l2, H * FF);
    A(L->mb_l2, H); A(L->vb_l2, H);
    A(L->mn1w, H); A(L->vn1w, H); A(L->mn1b, H); A(L->vn1b, H);
    A(L->mn2w, H); A(L->vn2w, H); A(L->mn2b, H); A(L->vn2b, H);
#undef A
    return ANN_OK;
}

static void free_layer(EncLayer *L)
{
#define F(p) hfree(p)
    F(L->W_qkv); F(L->b_qkv); F(L->W_out); F(L->b_out);
    F(L->W_l1); F(L->b_l1); F(L->W_l2); F(L->b_l2);
    F(L->n1w); F(L->n1b); F(L->n2w); F(L->n2b);
    F(L->gW_qkv); F(L->gb_qkv); F(L->gW_out); F(L->gb_out);
    F(L->gW_l1); F(L->gb_l1); F(L->gW_l2); F(L->gb_l2);
    F(L->gn1w); F(L->gn1b); F(L->gn2w); F(L->gn2b);
    F(L->mW_qkv); F(L->vW_qkv); F(L->mb_qkv); F(L->vb_qkv);
    F(L->mW_out); F(L->vW_out); F(L->mb_out); F(L->vb_out);
    F(L->mW_l1); F(L->vW_l1); F(L->mb_l1); F(L->vb_l1);
    F(L->mW_l2); F(L->vW_l2); F(L->mb_l2); F(L->vb_l2);
    F(L->mn1w); F(L->vn1w); F(L->mn1b); F(L->vn1b);
    F(L->mn2w); F(L->vn2w); F(L->mn2b); F(L->vn2b);
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

static void fill_ones(float *h, int n)
{
    for (int i = 0; i < n; i++) h[i] = 1.f;
}

/* ---- GEMM helpers (CBLAS, ColMajor mirror of cuBLAS calls) ---- */
/* Y(B,out) = X(B,in) @ W(out,in)^T */
static int gemm_xt(int B, int out, int in, const float *X, const float *W,
                   float *Y)
{
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, out, B, in, 1.f, W, in,
                X, in, 0.f, Y, out);
    return ANN_OK;
}

/* dX(B,in) = dY(B,out) @ W(out,in) */
static int gemm_dx(int B, int out, int in, const float *dY, const float *W,
                   float *dX)
{
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, in, B, out, 1.f, W,
                in, dY, out, 0.f, dX, in);
    return ANN_OK;
}

/* dW(out,in) += dY(B,out)^T @ X(B,in) */
static int gemm_dw(int B, int out, int in, const float *dY, const float *X,
                   float *dW)
{
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, in, out, B, 1.f, X, in,
                dY, out, 1.f, dW, in);
    return ANN_OK;
}

/* ---- CPU element-wise kernels (mirror model.cu k_add_bias/k_bias_grad/k_axpy) ---- */
static void add_bias(float *y, const float *b, int B, int D)
{
    for (int i = 0; i < B * D; i++) y[i] += b[i % D];
}

static void bias_grad(const float *dy, float *db, int B, int D)
{
    for (int d = 0; d < D; d++) {
        float s = 0.f;
        for (int i = 0; i < B; i++) s += dy[i * D + d];
        db[d] += s;
    }
}

static void axpy(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i++) y[i] += x[i];
}

/* ---- Linear layer fwd/bwd (mirror model.cu linear_fwd/linear_bwd ~472-507) ---- */
static int linear_fwd(const float *X, const float *W, const float *b, float *Y,
                      int B, int out, int in)
{
    if (gemm_xt(B, out, in, X, W, Y) != ANN_OK) return ANN_ERR;
    add_bias(Y, b, B, out);
    return ANN_OK;
}

static int linear_bwd(const float *X, const float *W, const float *dY, float *dX,
                      float *dW, float *db, int B, int out, int in, int acc_dx)
{
    if (dX) {
        if (acc_dx) {
            float *tmp = halloc((size_t)B * in);
            if (!tmp) return ANN_ERR;
            if (gemm_dx(B, out, in, dY, W, tmp) != ANN_OK) {
                hfree(tmp);
                return ANN_ERR;
            }
            axpy(dX, tmp, B * in);
            hfree(tmp);
        } else if (gemm_dx(B, out, in, dY, W, dX) != ANN_OK) {
            return ANN_ERR;
        }
    }
    if (gemm_dw(B, out, in, dY, X, dW) != ANN_OK) return ANN_ERR;
    bias_grad(dY, db, B, out);
    return ANN_OK;
}

/* ---- CPU mirrors of model.cu __global__ kernels (same math, same seed mixing) ---- */
static void add(float *y, const float *a, const float *b, int n)
{
    for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
}

static void add_pe(float *y, const float *x, const float *pe, int B, int H)
{
    for (int i = 0; i < B * H; i++) y[i] = x[i] + pe[i % H];
}

static void relu(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i++) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

static void relu_bwd(float *dx, const float *x, const float *dy, int n)
{
    for (int i = 0; i < n; i++) dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}

static void zero(float *x, int n)
{
    memset(x, 0, (size_t)n * sizeof(float));
}

static void scale(float *g, int n, float s)
{
    for (int i = 0; i < n; i++) g[i] *= s;
}

/* model.cu k_l1_bwd: dpred[i] = sign(pred-tgt)/n; returns sum(|pred-tgt|)
 * (caller divides by n, matching model_forward_bwd in model.cu). */
static float l1_bwd(const float *pred, const float *tgt, float *dpred, int n)
{
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - tgt[i];
        acc += fabsf(d);
        dpred[i] = (d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f)) / (float)n;
    }
    return acc;
}

/* Row-wise LayerNorm over H (model.cu k_ln_fwd; rsqrtf -> 1/sqrtf). */
static void ln_fwd(const float *x, float *y, const float *w, const float *b,
                   float *mean, float *rstd, int B, int H, float eps)
{
    for (int row = 0; row < B; row++) {
        const float *xr = x + (size_t)row * H;
        float m = 0.f;
        for (int i = 0; i < H; i++) m += xr[i];
        m /= (float)H;
        mean[row] = m;
        float v = 0.f;
        for (int i = 0; i < H; i++) {
            float d = xr[i] - m;
            v += d * d;
        }
        float rs = 1.f / sqrtf(v / (float)H + eps);
        rstd[row] = rs;
        float *yr = y + (size_t)row * H;
        for (int i = 0; i < H; i++) yr[i] = (xr[i] - m) * rs * w[i] + b[i];
    }
}

/* model.cu k_ln_bwd (atomicAdd on dw/db -> plain +=, rows processed serially). */
static void ln_bwd(const float *x, const float *dy, const float *w,
                   const float *mean, const float *rstd, float *dx,
                   float *dw, float *db, int B, int H)
{
    for (int row = 0; row < B; row++) {
        const float *xr = x + (size_t)row * H;
        const float *dyr = dy + (size_t)row * H;
        float m = mean[row], rs = rstd[row];
        float dnorm_sum = 0.f, dnorm_xsum = 0.f;
        for (int i = 0; i < H; i++) {
            float xhat = (xr[i] - m) * rs;
            float dyw = dyr[i] * w[i];
            dnorm_sum += dyw;
            dnorm_xsum += dyw * xhat;
            dw[i] += dyr[i] * xhat;
            db[i] += dyr[i];
        }
        float *dxr = dx + (size_t)row * H;
        for (int i = 0; i < H; i++) {
            float xhat = (xr[i] - m) * rs;
            float dyw = dyr[i] * w[i];
            dxr[i] = rs * (dyw - dnorm_sum / (float)H - xhat * dnorm_xsum / (float)H);
        }
    }
}

static void dropout(float *y, uint8_t *mask, const float *x, int n,
                    float drop_p, uint32_t seed, int train)
{
    for (int i = 0; i < n; i++) {
        if (!train || drop_p <= 0.f) {
            y[i] = x[i];
            if (mask) mask[i] = 1;
            continue;
        }
        uint32_t h = (uint32_t)i * 747796405u + seed;
        h ^= h >> 17;
        h *= 2246822519u;
        float u = (h >> 8) * (1.f / 16777216.f);
        uint8_t keep = u >= drop_p;
        mask[i] = keep;
        y[i] = keep ? x[i] / (1.f - drop_p) : 0.f;
    }
}

static void dropout_bwd(float *dx, const uint8_t *mask, const float *dy,
                        int n, float drop_p, int train)
{
    for (int i = 0; i < n; i++) {
        if (!train || drop_p <= 0.f) {
            dx[i] = dy[i];
            continue;
        }
        dx[i] = mask[i] ? dy[i] / (1.f - drop_p) : 0.f;
    }
}

static void copy_v(float *V, const float *qkv, int B, int H)
{
    for (int i = 0; i < B * H; i++) {
        int row = i / H;
        int col = i % H;
        V[i] = qkv[(size_t)row * 3 * H + 2 * H + col];
    }
}

static void scatter_v(float *qkv, const float *dV, int B, int H)
{
    for (int i = 0; i < B * H; i++) {
        int row = i / H;
        int col = i % H;
        qkv[(size_t)row * 3 * H + 2 * H + col] = dV[i];
    }
}

/* PyTorch MHA dropout on attn weights; seq_len=1 -> per-head scale of V
 * (model.cu k_head_drop). */
static void head_drop(float *V, uint8_t *mask, int B, int nhead, int hd,
                      float drop_p, uint32_t seed, int train)
{
    int nmask = B * nhead;
    for (int idx = 0; idx < nmask; idx++) {
        uint8_t keep = 1;
        float s = 1.f;
        if (train && drop_p > 0.f) {
            uint32_t h = (uint32_t)idx * 747796405u + seed;
            h ^= h >> 17;
            h *= 2246822519u;
            float u = (h >> 8) * (1.f / 16777216.f);
            keep = u >= drop_p;
            s = keep ? 1.f / (1.f - drop_p) : 0.f;
        }
        mask[idx] = keep;
        float *row = V + (size_t)(idx / nhead) * (nhead * hd) +
                     (size_t)(idx % nhead) * hd;
        for (int i = 0; i < hd; i++) row[i] *= s;
    }
}

static void head_drop_bwd(float *dV, const uint8_t *mask, int B, int nhead,
                          int hd, float drop_p, int train)
{
    int H = nhead * hd;
    if (!train || drop_p <= 0.f) return;
    for (int i = 0; i < B * H; i++) {
        int row = i / H;
        int col = i % H;
        int head = col / hd;
        dV[i] = mask[row * nhead + head] ? dV[i] / (1.f - drop_p) : 0.f;
    }
}

static void adamw(float *p, const float *g, float *m, float *v, int n,
                  float lr, float wd, float b1, float b2, float eps,
                  float b1t, float b2t)
{
    for (int i = 0; i < n; i++) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.f - b1) * gi;
        float vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mh = mi / (1.f - b1t);
        float vh = vi / (1.f - b2t);
        p[i] = p[i] - lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
    }
}

static float sqsum(const float *g, int n)
{
    float acc = 0.f;
    for (int i = 0; i < n; i++) acc += g[i] * g[i];
    return acc;
}

/* ---- Encoder fwd/bwd (model.cu enc_fwd ~904-940, enc_bwd ~942-991) ----
 * Simplified attention path kept: QKV linear -> copy V -> head_drop ->
 * out proj -> dropout -> residual+LN -> FFN(ReLU)+dropout -> residual+LN. */
static int enc_fwd(Work *wk, EncLayer *L, const float *src, float *dst,
                   int li, int B, int H, int FF, float drop_p, int train,
                   uint32_t seed)
{
    if (linear_fwd(src, L->W_qkv, L->b_qkv, wk->qkv[li], B, 3 * H, H) != ANN_OK)
        return ANN_ERR;
    copy_v(wk->V[li], wk->qkv[li], B, H);
    head_drop(wk->V[li], wk->m_aw[li], B, ANN_NHEAD, ANN_HEAD_DIM, drop_p,
              seed + 3u, train);
    if (linear_fwd(wk->V[li], L->W_out, L->b_out, wk->attn[li], B, H, H) != ANN_OK)
        return ANN_ERR;
    dropout(wk->res1[li], wk->m_attn[li], wk->attn[li], B * H, drop_p,
            seed + 17u, train);
    add(wk->ln1x[li], src, wk->res1[li], B * H);
    ln_fwd(wk->ln1x[li], wk->ln1y[li], L->n1w, L->n1b, wk->mean1[li],
           wk->rstd1[li], B, H, ANN_LN_EPS);
    if (linear_fwd(wk->ln1y[li], L->W_l1, L->b_l1, wk->ff_pre[li], B, FF, H) != ANN_OK)
        return ANN_ERR;
    relu(wk->relu[li], wk->ff_pre[li], B * FF);
    dropout(wk->ff_dropin[li], wk->m_ffmid[li], wk->relu[li], B * FF, drop_p,
            seed + 1000u, train);
    if (linear_fwd(wk->ff_dropin[li], L->W_l2, L->b_l2, wk->ff_out[li], B, H, FF) != ANN_OK)
        return ANN_ERR;
    dropout(wk->res2[li], wk->m_ff[li], wk->ff_out[li], B * H, drop_p,
            seed + 2000u, train);
    add(wk->ln2x[li], wk->ln1y[li], wk->res2[li], B * H);
    ln_fwd(wk->ln2x[li], dst, L->n2w, L->n2b, wk->mean2[li], wk->rstd2[li],
           B, H, ANN_LN_EPS);
    return ANN_OK;
}

static int enc_bwd(Work *wk, EncLayer *L, const float *src,
                   const float *d_dst, float *d_src, int li, int B, int H,
                   int FF, float drop_p, int train)
{
    /* Buffer reuse mirrors model.cu exactly (attn/ff_out/res1/ln1y/res2). */
    float *d_ln2x = wk->attn[li];
    ln_bwd(wk->ln2x[li], d_dst, L->n2w, wk->mean2[li], wk->rstd2[li],
           d_ln2x, L->gn2w, L->gn2b, B, H);

    float *d_ffdrop = wk->ff_out[li];
    dropout_bwd(d_ffdrop, wk->m_ff[li], d_ln2x, B * H, drop_p, train);
    if (linear_bwd(wk->ff_dropin[li], L->W_l2, d_ffdrop, wk->ws_ff, L->gW_l2,
                   L->gb_l2, B, H, FF, 0) != ANN_OK)
        return ANN_ERR;
    dropout_bwd(wk->ws_ff, wk->m_ffmid[li], wk->ws_ff, B * FF, drop_p, train);
    relu_bwd(wk->ws_ff, wk->relu[li], wk->ws_ff, B * FF);

    float *d_src1n = wk->res1[li];
    if (linear_bwd(wk->ln1y[li], L->W_l1, wk->ws_ff, d_src1n, L->gW_l1,
                   L->gb_l1, B, FF, H, 0) != ANN_OK)
        return ANN_ERR;
    axpy(d_src1n, d_ln2x, B * H);

    float *d_ln1x = wk->ln1y[li];
    ln_bwd(wk->ln1x[li], d_src1n, L->n1w, wk->mean1[li], wk->rstd1[li],
           d_ln1x, L->gn1w, L->gn1b, B, H);
    memcpy(d_src, d_ln1x, (size_t)B * H * sizeof(float));

    float *d_attn = wk->d_scratch;
    dropout_bwd(d_attn, wk->m_attn[li], d_ln1x, B * H, drop_p, train);
    float *d_V = wk->res2[li];
    if (linear_bwd(wk->V[li], L->W_out, d_attn, d_V, L->gW_out, L->gb_out,
                   B, H, H, 0) != ANN_OK)
        return ANN_ERR;
    head_drop_bwd(d_V, wk->m_aw[li], B, ANN_NHEAD, ANN_HEAD_DIM, drop_p, train);
    zero(wk->qkv[li], B * 3 * H);
    scatter_v(wk->qkv[li], d_V, B, H);
    if (linear_bwd(src, L->W_qkv, wk->qkv[li], d_src, L->gW_qkv, L->gb_qkv,
                   B, 3 * H, H, 1) != ANN_OK)
        return ANN_ERR;
    return ANN_OK;
}

/* model.cu net_forward (~1033-1055). */
static int net_forward(Model *m, int B, uint32_t seed)
{
    Work *wk = (Work *)m->work;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    int train = m->training;
    float drop_p = m->dropout;
    if (linear_fwd(wk->x_buf, m->W_emb, m->b_emb, wk->h[0], B, H, I) != ANN_OK)
        return ANN_ERR;
    add_pe(wk->h[0], wk->h[0], m->pe0, B, H);
    for (int li = 0; li < m->nlayers; li++) {
        if (enc_fwd(wk, &m->layers[li], wk->h[li], wk->h[li + 1], li, B, H, FF,
                    drop_p, train, seed + (uint32_t)li * 4096u) != ANN_OK)
            return ANN_ERR;
    }
    if (linear_fwd(wk->h[m->nlayers], m->W_fc, m->b_fc, wk->pred, B, O, H) != ANN_OK)
        return ANN_ERR;
    return ANN_OK;
}

typedef struct {
    float *p, *g, *m, *v;
    int n;
} PSlot;

/* model.cu collect_params (~998-1031). */
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

/* model.cu gather_batch (~1165-1173). */
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

int cpu_runtime_ok(void) { return 1; }

int cpu_harden_reset(const char *why)
{
    /* No global/static workspace on the CPU port — every Model owns its
     * own Work, so there is nothing to invalidate here. Callers should
     * model_destroy() the failing Model and start over. */
    (void)why;
    return ANN_OK;
}

/* model_create / model_destroy / xavier / zero_grad implemented in Task 3.
 * Forward/train functions implemented in Task 5 (below). */
int model_create(Model *m, const char *code, int dropout_enable)
{
    if (!m || !code) return ANN_ERR;
    memset(m, 0, sizeof(*m));
    snprintf(m->code, sizeof(m->code), "%s", code);
    m->insize = ANN_INSIZE;
    m->hidden = ANN_HIDDEN;
    m->outsize = ANN_OUTSIZE;
    m->nhead = ANN_NHEAD;
    m->nlayers = ANN_NLAYERS;
    m->d_ff = ANN_D_FF;
    m->dropout = dropout_enable ? ANN_DROPOUT : 0.f;
    m->training = 0;
    m->interrupt = 0;
    m->adam_step = 0;

    if (!cpu_runtime_ok()) {
        fprintf(stderr, "cpu_runtime_ok failed\n");
        return ANN_ERR;
    }

    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;

    m->W_emb = halloc((size_t)H * I);
    m->b_emb = halloc((size_t)H);
    m->W_fc = halloc((size_t)O * H);
    m->b_fc = halloc((size_t)O);
    m->gW_emb = halloc((size_t)H * I);
    m->gb_emb = halloc((size_t)H);
    m->gW_fc = halloc((size_t)O * H);
    m->gb_fc = halloc((size_t)O);
    m->mW_emb = halloc((size_t)H * I);
    m->vW_emb = halloc((size_t)H * I);
    m->mb_emb = halloc((size_t)H);
    m->vb_emb = halloc((size_t)H);
    m->mW_fc = halloc((size_t)O * H);
    m->vW_fc = halloc((size_t)O * H);
    m->mb_fc = halloc((size_t)O);
    m->vb_fc = halloc((size_t)O);
    m->pe0 = halloc((size_t)H);
    if (!m->W_emb || !m->W_fc || !m->pe0) {
        model_destroy(m);
        return ANN_ERR;
    }

    for (int i = 0; i < ANN_NLAYERS; i++) {
        if (alloc_layer(&m->layers[i], H, FF) != ANN_OK) {
            model_destroy(m);
            return ANN_ERR;
        }
    }

    m->work = (struct Work *)calloc(1, sizeof(Work));
    if (!m->work) {
        model_destroy(m);
        return ANN_ERR;
    }
    Work *wk = (Work *)m->work;
    if (io_alloc(wk, ANN_MAX_BATCH, I, H, O) != ANN_OK ||
        work_alloc(wk, ANN_MAX_BATCH, H, FF) != ANN_OK) {
        model_destroy(m);
        return ANN_ERR;
    }

    /* pe0: even indices 0, odd indices 1 — matches CUDA host staging. */
    for (int i = 0; i < H; i++) {
        m->pe0[i] = (i % 2 == 0) ? 0.f : 1.f;
    }

    model_xavier_init(m, ANN_SEED);
    return ANN_OK;
}

void model_destroy(Model *m)
{
    if (!m) return;
    if (m->work) {
        work_free((Work *)m->work);
        free(m->work);
        m->work = NULL;
    }
    hfree(m->W_emb); hfree(m->b_emb); hfree(m->W_fc); hfree(m->b_fc);
    hfree(m->gW_emb); hfree(m->gb_emb); hfree(m->gW_fc); hfree(m->gb_fc);
    hfree(m->mW_emb); hfree(m->vW_emb); hfree(m->mb_emb); hfree(m->vb_emb);
    hfree(m->mW_fc); hfree(m->vW_fc); hfree(m->mb_fc); hfree(m->vb_fc);
    hfree(m->pe0);
    for (int i = 0; i < ANN_NLAYERS; i++) {
        free_layer(&m->layers[i]);
    }
    memset(m, 0, sizeof(*m));
}

void model_xavier_init(Model *m, uint32_t seed)
{
    if (!m) return;
    uint32_t rng = seed ? seed : 1u;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    host_xavier(m->W_emb, H, I, &rng);
    host_xavier(m->W_fc, O, H, &rng);
    memset(m->b_emb, 0, (size_t)H * sizeof(float));
    memset(m->b_fc, 0, (size_t)O * sizeof(float));
    for (int i = 0; i < m->nlayers; i++) {
        EncLayer *L = &m->layers[i];
        host_xavier(L->W_qkv, 3 * H, H, &rng);
        host_xavier(L->W_out, H, H, &rng);
        host_xavier(L->W_l1, FF, H, &rng);
        host_xavier(L->W_l2, H, FF, &rng);
        memset(L->b_qkv, 0, (size_t)3 * H * sizeof(float));
        memset(L->b_out, 0, (size_t)H * sizeof(float));
        memset(L->b_l1, 0, (size_t)FF * sizeof(float));
        memset(L->b_l2, 0, (size_t)H * sizeof(float));
        fill_ones(L->n1w, H);
        fill_ones(L->n2w, H);
        memset(L->n1b, 0, (size_t)H * sizeof(float));
        memset(L->n2b, 0, (size_t)H * sizeof(float));
    }
}

void model_zero_grad(Model *m)
{
    if (!m) return;
    int H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
    memset(m->gW_emb, 0, (size_t)H * I * sizeof(float));
    memset(m->gb_emb, 0, (size_t)H * sizeof(float));
    memset(m->gW_fc, 0, (size_t)O * H * sizeof(float));
    memset(m->gb_fc, 0, (size_t)O * sizeof(float));
    for (int i = 0; i < m->nlayers; i++) {
        EncLayer *L = &m->layers[i];
        memset(L->gW_qkv, 0, (size_t)3 * H * H * sizeof(float));
        memset(L->gb_qkv, 0, (size_t)3 * H * sizeof(float));
        memset(L->gW_out, 0, (size_t)H * H * sizeof(float));
        memset(L->gb_out, 0, (size_t)H * sizeof(float));
        memset(L->gW_l1, 0, (size_t)FF * H * sizeof(float));
        memset(L->gb_l1, 0, (size_t)FF * sizeof(float));
        memset(L->gW_l2, 0, (size_t)H * FF * sizeof(float));
        memset(L->gb_l2, 0, (size_t)H * sizeof(float));
        memset(L->gn1w, 0, (size_t)H * sizeof(float));
        memset(L->gn1b, 0, (size_t)H * sizeof(float));
        memset(L->gn2w, 0, (size_t)H * sizeof(float));
        memset(L->gn2b, 0, (size_t)H * sizeof(float));
    }
}

/* model.cu model_forward (~1057-1075): eval mode, memcpy in/out. */
int model_forward(Model *m, const float *x_host, int batch, float *y_host)
{
    if (!m || !x_host || !y_host) return ANN_ERR;
    if (batch <= 0 || batch > ANN_MAX_BATCH) return ANN_ERR;
    Work *wk = (Work *)m->work;
    int saved = m->training;
    m->training = 0;
    memcpy(wk->x_buf, x_host, (size_t)batch * m->insize * sizeof(float));
    if (net_forward(m, batch, 0) != ANN_OK) {
        m->training = saved;
        return ANN_ERR;
    }
    memcpy(y_host, wk->pred, (size_t)batch * m->outsize * sizeof(float));
    m->training = saved;
    return ANN_OK;
}

/* model.cu model_forward_bwd (~1077-1116). */
int model_forward_bwd(Model *m, const float *x_host, const float *y_host,
                      int batch, float *loss_out)
{
    if (!m || !x_host || !y_host) return ANN_ERR;
    if (batch <= 0 || batch > ANN_MAX_BATCH) return ANN_ERR;
    Work *wk = (Work *)m->work;
    int H = m->hidden, I = m->insize, O = m->outsize;
    int n = batch * O;
    memcpy(wk->x_buf, x_host, (size_t)batch * I * sizeof(float));
    memcpy(wk->y_buf, y_host, (size_t)batch * O * sizeof(float));
    uint32_t seed = (uint32_t)m->adam_step * 10007u + 1u;
    if (net_forward(m, batch, seed) != ANN_OK) return ANN_ERR;
    float loss_sum = l1_bwd(wk->pred, wk->y_buf, wk->dpred, n);
    if (loss_out) *loss_out = loss_sum / (float)n;
    if (linear_bwd(wk->h[m->nlayers], m->W_fc, wk->dpred, wk->d_h[m->nlayers],
                   m->gW_fc, m->gb_fc, batch, O, H, 0) != ANN_OK)
        return ANN_ERR;
    for (int li = m->nlayers - 1; li >= 0; li--) {
        if (enc_bwd(wk, &m->layers[li], wk->h[li], wk->d_h[li + 1], wk->d_h[li],
                    li, batch, H, m->d_ff, m->dropout, m->training) != ANN_OK)
            return ANN_ERR;
    }
    if (linear_bwd(wk->x_buf, m->W_emb, wk->d_h[0], NULL, m->gW_emb, m->gb_emb,
                   batch, H, I, 0) != ANN_OK)
        return ANN_ERR;
    return ANN_OK;
}

/* model.cu model_adamw_step (~1118-1135). */
int model_adamw_step(Model *m, float lr, float weight_decay)
{
    PSlot slots[64];
    int n = collect_params(m, slots);
    m->adam_step += 1;
    float b1t = 1.f, b2t = 1.f;
    for (int t = 0; t < m->adam_step; t++) {
        b1t *= ANN_ADAM_B1;
        b2t *= ANN_ADAM_B2;
    }
    for (int i = 0; i < n; i++) {
        adamw(slots[i].p, slots[i].g, slots[i].m, slots[i].v, slots[i].n, lr,
              weight_decay, ANN_ADAM_B1, ANN_ADAM_B2, ANN_ADAM_EPS, b1t, b2t);
    }
    return ANN_OK;
}

/* model.cu model_clip_grad (~1137-1154). */
int model_clip_grad(Model *m, float max_norm)
{
    PSlot slots[64];
    int n = collect_params(m, slots);
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += sqsum(slots[i].g, slots[i].n);
    float total = sqrtf(ss);
    if (total > max_norm) {
        float s = max_norm / (total + 1e-6f);
        for (int i = 0; i < n; i++) scale(slots[i].g, slots[i].n, s);
    }
    return ANN_OK;
}

float model_cosine_lr(float base_lr, int ep, int tmax)
{
    /* must match c_annfund/src/model.cu */
    if (ep <= 1 || tmax <= 0) {
        return base_lr;
    }
    float t = (float)(ep - 1) / (float)tmax;
    return 0.5f * base_lr * (1.f + cosf(3.14159265358979323846f * t));
}

/* model.cu model_train (~1175-1298): same loop, UDP log, save cadence,
 * loss threshold. */
int model_train(Model *m, const Dataset *ds, int epoch, float lr,
                float loss_threshold, int shuffle)
{
    if (!ds || !ds->has_y || ds->n <= 0 || epoch <= 0) return ANN_ERR;
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
        if (m->interrupt) break;
        if (shuffle) {
            dataset_shuffle_idx(idx, ds->n, &rng);
        } else {
            for (int i = 0; i < ds->n; i++) idx[i] = i;
        }
        loss_one = 0.f;
        float lr_now = model_cosine_lr(lr, ep, epoch);
        for (int start = 0; start < ds->n; start += ANN_MAX_BATCH) {
            if (m->interrupt) break;
            int B = ds->n - start;
            if (B > ANN_MAX_BATCH) B = ANN_MAX_BATCH;
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
        if (ep % 1000 == 0) {
            float avg = 0.f;
            for (int i = 0; i < nhist; i++) avg += hist[i];
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
        if (ep > 1000 && ep % 20000 == 1) {
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

/* model.cu model_predict (~1300-1321): batch loop, training=0. */
int model_predict(Model *m, const Dataset *ds, float *pred_out)
{
    if (!ds || ds->n <= 0 || !pred_out) return ANN_ERR;
    int saved = m->training;
    m->training = 0;
    for (int start = 0; start < ds->n; start += ANN_MAX_BATCH) {
        int B = ds->n - start;
        if (B > ANN_MAX_BATCH) B = ANN_MAX_BATCH;
        if (model_forward(m, ds->x + (size_t)start * ANN_INSIZE, B,
                          pred_out + (size_t)start * ANN_OUTSIZE) != ANN_OK) {
            m->training = saved;
            return ANN_ERR;
        }
    }
    m->training = saved;
    return ANN_OK;
}
