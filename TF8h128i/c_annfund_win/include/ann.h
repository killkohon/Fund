#ifndef C_ANNFUND_ANN_H
#define C_ANNFUND_ANN_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define ANN_INSIZE 128
#define ANN_HIDDEN 256
#define ANN_OUTSIZE 3
#define ANN_NHEAD 8
#define ANN_NLAYERS 2
#define ANN_D_FF 2048
#define ANN_HEAD_DIM (ANN_HIDDEN / ANN_NHEAD)
#define ANN_SERIAL_ISIZE 126
#define ANN_SERIAL_OSIZE 20
#define ANN_MAX_BATCH 360
#define ANN_NAV_TRAIN 800
#define ANN_NAV_PRED 200
#define ANN_DROPOUT 0.1f
#define ANN_LN_EPS 1e-5f
#define ANN_ADAM_B1 0.9f
#define ANN_ADAM_B2 0.999f
#define ANN_ADAM_EPS 1e-8f
#define ANN_WD 1e-4f
#define ANN_CLIP 1.0f
#define ANN_SEED 42u
#define ANN_CODE_LEN 6
#define ANN_MAX_CODES 2048

#define ANN_OK 0
#define ANN_ERR -1

static inline void ann_log(const char *msg) { fprintf(stderr, "%s\n", msg); }

#endif
