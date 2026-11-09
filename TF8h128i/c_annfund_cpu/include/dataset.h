#ifndef C_ANNFUND_DATASET_H
#define C_ANNFUND_DATASET_H

#include "ann.h"
#include "db.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *x; /* n * ANN_INSIZE */
    float *y; /* n * ANN_OUTSIZE, train only */
    int n;
    int has_y;
} Dataset;

int dataset_from_serial_train(const NavSeries *nav, Dataset *ds);
int dataset_from_serial_pred(const NavSeries *nav, Dataset *ds);
void dataset_free(Dataset *ds);
void dataset_shuffle_idx(int *idx, int n, uint32_t *rng);

#ifdef __cplusplus
}
#endif

#endif
