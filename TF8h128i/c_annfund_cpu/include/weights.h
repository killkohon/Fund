#ifndef C_ANNFUND_WEIGHTS_H
#define C_ANNFUND_WEIGHTS_H

#include "ann.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Model;

int weights_save(const struct Model *m, const char *path);
int weights_load(struct Model *m, const char *path);
void weights_bin_path(const char *code, char *out, size_t n);
int weights_export_pth(const char *binpath);
int weights_import_pth(const char *pthpath);
int weights_sync_bin(const char *code);

#ifdef __cplusplus
}
#endif

#endif
