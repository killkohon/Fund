#ifndef C_ANNFUND_MODEL_H
#define C_ANNFUND_MODEL_H

#include "ann.h"
#include "dataset.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EncLayer {
    float *W_qkv, *b_qkv;       /* (3H, H), (3H,) */
    float *W_out, *b_out;       /* (H, H), (H,) */
    float *W_l1, *b_l1;         /* (FF, H), (FF,) */
    float *W_l2, *b_l2;         /* (H, FF), (H,) */
    float *n1w, *n1b, *n2w, *n2b;

    float *gW_qkv, *gb_qkv, *gW_out, *gb_out;
    float *gW_l1, *gb_l1, *gW_l2, *gb_l2;
    float *gn1w, *gn1b, *gn2w, *gn2b;

    float *mW_qkv, *vW_qkv, *mb_qkv, *vb_qkv;
    float *mW_out, *vW_out, *mb_out, *vb_out;
    float *mW_l1, *vW_l1, *mb_l1, *vb_l1;
    float *mW_l2, *vW_l2, *mb_l2, *vb_l2;
    float *mn1w, *vn1w, *mn1b, *vn1b;
    float *mn2w, *vn2w, *mn2b, *vn2b;
} EncLayer;

/* Forward-declare Work; definition lives in model.c */
struct Work;

typedef struct Model {
    char code[8];
    int insize, hidden, outsize, nhead, nlayers, d_ff;
    float dropout;
    int training;
    volatile int interrupt;

    float *W_emb, *b_emb, *W_fc, *b_fc;
    float *gW_emb, *gb_emb, *gW_fc, *gb_fc;
    float *mW_emb, *vW_emb, *mb_emb, *vb_emb;
    float *mW_fc, *vW_fc, *mb_fc, *vb_fc;
    EncLayer layers[ANN_NLAYERS];
    float *pe0; /* position-0 encoding, hidden */

    /* workspace */
    float *ws;
    size_t ws_floats;
    uint8_t *drop_ws;
    size_t drop_bytes;

    int adam_step;
    /* NO cublas handle — host only */
    struct Work *work; /* per-model workspace; owned by Model */
} Model;

int cpu_runtime_ok(void);
int cpu_harden_reset(const char *why);

int model_create(Model *m, const char *code, int dropout_enable);
void model_destroy(Model *m);
void model_zero_grad(Model *m);
void model_xavier_init(Model *m, uint32_t seed);

int model_forward(Model *m, const float *x_host, int batch, float *y_host);
int model_forward_bwd(Model *m, const float *x_host, const float *y_host, int batch, float *loss_out);
int model_adamw_step(Model *m, float lr, float weight_decay);
int model_clip_grad(Model *m, float max_norm);
float model_cosine_lr(float base_lr, int ep, int tmax);

int model_train(Model *m, const Dataset *ds, int epoch, float lr, float loss_threshold, int shuffle);
int model_predict(Model *m, const Dataset *ds, float *pred_out /* n * 3 */);

#ifdef __cplusplus
}
#endif

#endif
