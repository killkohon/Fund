#include "weights.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define ANN_MAGIC 0x424E4E41u /* 'ANNB' */
#define ANN_BIN_VER 1u
#define ANN_NAME_LEN 64

typedef struct {
    char name[ANN_NAME_LEN];
    float *dev;
    int ndim;
    int dims[4];
} TensorRef;

static void fill_refs(const Model *m, TensorRef *t, int *n_out)
{
    int n = 0, H = m->hidden, I = m->insize, O = m->outsize, FF = m->d_ff;
#define T2(nm, ptr, d0, d1)                                                    \
    do {                                                                       \
        memset(&t[n], 0, sizeof(t[n]));                                        \
        snprintf(t[n].name, ANN_NAME_LEN, "%s", nm);                           \
        t[n].dev = (ptr);                                                      \
        t[n].ndim = 2;                                                         \
        t[n].dims[0] = (d0);                                                   \
        t[n].dims[1] = (d1);                                                   \
        n++;                                                                   \
    } while (0)
#define T1(nm, ptr, d0)                                                        \
    do {                                                                       \
        memset(&t[n], 0, sizeof(t[n]));                                        \
        snprintf(t[n].name, ANN_NAME_LEN, "%s", nm);                           \
        t[n].dev = (ptr);                                                      \
        t[n].ndim = 1;                                                         \
        t[n].dims[0] = (d0);                                                   \
        n++;                                                                   \
    } while (0)

    T2("embedding.weight", m->W_emb, H, I);
    T1("embedding.bias", m->b_emb, H);
    T2("fc.weight", m->W_fc, O, H);
    T1("fc.bias", m->b_fc, O);
    for (int i = 0; i < m->nlayers; i++) {
        const EncLayer *L = &m->layers[i];
        char nm[ANN_NAME_LEN];
        snprintf(nm, sizeof(nm), "transformer.layers.%d.self_attn.in_proj_weight", i);
        T2(nm, L->W_qkv, 3 * H, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.self_attn.in_proj_bias", i);
        T1(nm, L->b_qkv, 3 * H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.self_attn.out_proj.weight", i);
        T2(nm, L->W_out, H, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.self_attn.out_proj.bias", i);
        T1(nm, L->b_out, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.linear1.weight", i);
        T2(nm, L->W_l1, FF, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.linear1.bias", i);
        T1(nm, L->b_l1, FF);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.linear2.weight", i);
        T2(nm, L->W_l2, H, FF);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.linear2.bias", i);
        T1(nm, L->b_l2, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.norm1.weight", i);
        T1(nm, L->n1w, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.norm1.bias", i);
        T1(nm, L->n1b, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.norm2.weight", i);
        T1(nm, L->n2w, H);
        snprintf(nm, sizeof(nm), "transformer.layers.%d.norm2.bias", i);
        T1(nm, L->n2b, H);
    }
#undef T1
#undef T2
    *n_out = n;
}

static int numel(const TensorRef *t)
{
    int n = 1;
    for (int i = 0; i < t->ndim; i++) {
        n *= t->dims[i];
    }
    return n;
}

void weights_bin_path(const char *code, char *out, size_t n)
{
    snprintf(out, n, "./ann/%s.bin", code);
}

int weights_save(const Model *m, const char *path)
{
    TensorRef refs[64];
    int n = 0;
    fill_refs(m, refs, &n);
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        return ANN_ERR;
    }
    uint32_t magic = ANN_MAGIC, ver = ANN_BIN_VER, nt = (uint32_t)n;
    int32_t hdr[6] = {m->insize, m->hidden, m->outsize, m->nlayers, m->nhead,
                      m->d_ff};
    char code[8] = {0};
    snprintf(code, sizeof(code), "%s", m->code);
    fwrite(&magic, 4, 1, fp);
    fwrite(&ver, 4, 1, fp);
    fwrite(code, 8, 1, fp);
    fwrite(hdr, sizeof(int32_t), 6, fp);
    fwrite(&nt, 4, 1, fp);
    for (int i = 0; i < n; i++) {
        int32_t ndim = refs[i].ndim;
        int32_t dims[4] = {0};
        memcpy(dims, refs[i].dims, sizeof(int) * 4);
        int32_t count = numel(&refs[i]);
        int32_t nbytes = count * (int32_t)sizeof(float);
        fwrite(refs[i].name, ANN_NAME_LEN, 1, fp);
        fwrite(&ndim, 4, 1, fp);
        fwrite(dims, 4, 4, fp);
        fwrite(&nbytes, 4, 1, fp);
        fwrite(refs[i].dev, 1, (size_t)nbytes, fp);
    }
    fclose(fp);
    if (weights_export_pth(path) != ANN_OK) {
        fprintf(stderr, "weights_save: pth export failed for %s\n", path);
    }
    return ANN_OK;
}

int weights_load(Model *m, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return ANN_ERR;
    }
    uint32_t magic = 0, ver = 0, nt = 0;
    char code[8] = {0};
    int32_t hdr[6] = {0};
    if (fread(&magic, 4, 1, fp) != 1 || magic != ANN_MAGIC ||
        fread(&ver, 4, 1, fp) != 1 || ver != ANN_BIN_VER ||
        fread(code, 8, 1, fp) != 1 || fread(hdr, sizeof(int32_t), 6, fp) != 6 ||
        fread(&nt, 4, 1, fp) != 1) {
        fclose(fp);
        return ANN_ERR;
    }
    if (hdr[0] != m->insize || hdr[1] != m->hidden || hdr[2] != m->outsize ||
        hdr[3] != m->nlayers || hdr[5] != m->d_ff) {
        fprintf(stderr, "weight shape mismatch in %s\n", path);
        fclose(fp);
        return ANN_ERR;
    }
    TensorRef refs[64];
    int nref = 0;
    fill_refs(m, refs, &nref);
    for (uint32_t i = 0; i < nt; i++) {
        char name[ANN_NAME_LEN];
        int32_t ndim = 0, dims[4] = {0}, nbytes = 0;
        if (fread(name, ANN_NAME_LEN, 1, fp) != 1 || fread(&ndim, 4, 1, fp) != 1 ||
            fread(dims, 4, 4, fp) != 4 || fread(&nbytes, 4, 1, fp) != 1) {
            fclose(fp);
            return ANN_ERR;
        }
        name[ANN_NAME_LEN - 1] = 0;
        float *host = (float *)malloc((size_t)nbytes);
        if (!host || fread(host, 1, (size_t)nbytes, fp) != (size_t)nbytes) {
            free(host);
            fclose(fp);
            return ANN_ERR;
        }
        int found = 0;
        for (int j = 0; j < nref; j++) {
            if (strcmp(refs[j].name, name) == 0) {
                if (numel(&refs[j]) * (int)sizeof(float) != nbytes) {
                    fprintf(stderr, "size mismatch %s\n", name);
                    free(host);
                    fclose(fp);
                    return ANN_ERR;
                }
                memcpy(refs[j].dev, host, (size_t)nbytes);
                found = 1;
                break;
            }
        }
        free(host);
        if (!found) {
            fprintf(stderr, "unknown tensor %s (skipped)\n", name);
        }
    }
    fclose(fp);
    if (code[0] && m->code[0] == 0) {
        snprintf(m->code, sizeof(m->code), "%s", code);
    }
    return ANN_OK;
}
