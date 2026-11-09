#include "weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#endif

#define ANN_MAGIC 0x424E4E41u /* 'ANNB' */
#define ANN_BIN_VER 1u
#define ANN_NAME_LEN 64
#define PE_MAX_LEN 5000
#define PTH_MAX_TENSORS 80
#define ZIP_MAX_FILES 96

typedef struct {
    char name[ANN_NAME_LEN];
    int ndim;
    int dims[4];
    int count;
    float *data;
} PthTensor;

typedef struct {
    unsigned char *p;
    size_t n, cap;
} Buf;

typedef struct {
    char name[96];
    uint32_t crc, size, offset;
} ZipRec;

static int buf_reserve(Buf *b, size_t extra)
{
    size_t cap;
    unsigned char *p;
    if (b->n + extra <= b->cap) {
        return 0;
    }
    cap = b->cap ? b->cap : 256;
    while (cap < b->n + extra) {
        cap *= 2;
    }
    p = (unsigned char *)realloc(b->p, cap);
    if (!p) {
        return -1;
    }
    b->p = p;
    b->cap = cap;
    return 0;
}

static int buf_put(Buf *b, const void *s, size_t n)
{
    if (buf_reserve(b, n) != 0) {
        return -1;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    return 0;
}

static int buf_u8(Buf *b, unsigned char x)
{
    return buf_put(b, &x, 1);
}

static uint32_t crc32_of(const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    uint32_t crc = 0xFFFFFFFFu;
    size_t i;
    int b;
    for (i = 0; i < n; i++) {
        crc ^= p[i];
        for (b = 0; b < 8; b++) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

static void wr_u16(FILE *fp, uint16_t v)
{
    unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
    fwrite(b, 1, 2, fp);
}

static void wr_u32(FILE *fp, uint32_t v)
{
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                          (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    fwrite(b, 1, 4, fp);
}

static int zip_add(FILE *fp, ZipRec *recs, int *nrec, const char *name,
                   const void *data, uint32_t size)
{
    ZipRec *r;
    long off;
    size_t nlen;
    if (*nrec >= ZIP_MAX_FILES) {
        return -1;
    }
    nlen = strlen(name);
    if (nlen == 0 || nlen >= sizeof(recs[0].name)) {
        return -1;
    }
    off = ftell(fp);
    if (off < 0) {
        return -1;
    }
    r = &recs[*nrec];
    memset(r, 0, sizeof(*r));
    memcpy(r->name, name, nlen + 1);
    r->crc = crc32_of(data, size);
    r->size = size;
    r->offset = (uint32_t)off;
    wr_u32(fp, 0x04034b50u);
    wr_u16(fp, 20);
    wr_u16(fp, 0);
    wr_u16(fp, 0);
    wr_u16(fp, 0);
    wr_u16(fp, 0);
    wr_u32(fp, r->crc);
    wr_u32(fp, size);
    wr_u32(fp, size);
    wr_u16(fp, (uint16_t)nlen);
    wr_u16(fp, 0);
    if (fwrite(name, 1, nlen, fp) != nlen) {
        return -1;
    }
    if (size > 0 && fwrite(data, 1, size, fp) != size) {
        return -1;
    }
    (*nrec)++;
    return 0;
}

static int zip_finish(FILE *fp, const ZipRec *recs, int nrec)
{
    long cd_off;
    uint32_t cd_size;
    int i;
    cd_off = ftell(fp);
    if (cd_off < 0) {
        return -1;
    }
    for (i = 0; i < nrec; i++) {
        size_t nlen = strlen(recs[i].name);
        wr_u32(fp, 0x02014b50u);
        wr_u16(fp, 20);
        wr_u16(fp, 20);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u32(fp, recs[i].crc);
        wr_u32(fp, recs[i].size);
        wr_u32(fp, recs[i].size);
        wr_u16(fp, (uint16_t)nlen);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u16(fp, 0);
        wr_u32(fp, 0);
        wr_u32(fp, recs[i].offset);
        if (fwrite(recs[i].name, 1, nlen, fp) != nlen) {
            return -1;
        }
    }
    cd_size = (uint32_t)(ftell(fp) - cd_off);
    wr_u32(fp, 0x06054b50u);
    wr_u16(fp, 0);
    wr_u16(fp, 0);
    wr_u16(fp, (uint16_t)nrec);
    wr_u16(fp, (uint16_t)nrec);
    wr_u32(fp, cd_size);
    wr_u32(fp, (uint32_t)cd_off);
    wr_u16(fp, 0);
    return ferror(fp) ? -1 : 0;
}

static int pickle_int(Buf *b, int v)
{
    unsigned char tmp[5];
    if (v >= 0 && v <= 255) {
        tmp[0] = 0x4b;
        tmp[1] = (unsigned char)v;
        return buf_put(b, tmp, 2);
    }
    if (v >= 0 && v <= 65535) {
        tmp[0] = 0x4d;
        tmp[1] = (unsigned char)v;
        tmp[2] = (unsigned char)(v >> 8);
        return buf_put(b, tmp, 3);
    }
    tmp[0] = 0x4a;
    tmp[1] = (unsigned char)v;
    tmp[2] = (unsigned char)(v >> 8);
    tmp[3] = (unsigned char)(v >> 16);
    tmp[4] = (unsigned char)(v >> 24);
    return buf_put(b, tmp, 5);
}

static int pickle_str(Buf *b, const char *s)
{
    size_t n = strlen(s);
    unsigned char hdr[5];
    hdr[0] = 0x58;
    hdr[1] = (unsigned char)n;
    hdr[2] = (unsigned char)(n >> 8);
    hdr[3] = (unsigned char)(n >> 16);
    hdr[4] = (unsigned char)(n >> 24);
    if (buf_put(b, hdr, 5) != 0) {
        return -1;
    }
    return buf_put(b, s, n);
}

static int pickle_global(Buf *b, const char *mod, const char *name)
{
    if (buf_u8(b, 0x63) != 0) {
        return -1;
    }
    if (buf_put(b, mod, strlen(mod)) != 0 || buf_u8(b, '\n') != 0) {
        return -1;
    }
    return buf_put(b, name, strlen(name)) != 0 || buf_u8(b, '\n');
}

static int pickle_tuple(Buf *b, const int *v, int n)
{
    int i;
    if (n <= 0) {
        return buf_u8(b, 0x29);
    }
    if (n > 3) {
        if (buf_u8(b, 0x28) != 0) {
            return -1;
        }
        for (i = 0; i < n; i++) {
            if (pickle_int(b, v[i]) != 0) {
                return -1;
            }
        }
        return buf_u8(b, 0x74);
    }
    for (i = 0; i < n; i++) {
        if (pickle_int(b, v[i]) != 0) {
            return -1;
        }
    }
    return buf_u8(b, (unsigned char)(0x84 + n));
}

static int pickle_tensor(Buf *b, const PthTensor *t, int storage_id)
{
    char key[16];
    int stride[4];
    int i, s;
    snprintf(key, sizeof(key), "%d", storage_id);
    s = 1;
    for (i = t->ndim - 1; i >= 0; i--) {
        stride[i] = s;
        s *= t->dims[i];
    }
    if (pickle_str(b, t->name) != 0) {
        return -1;
    }
    if (pickle_global(b, "torch._utils", "_rebuild_tensor_v2") != 0) {
        return -1;
    }
    if (buf_u8(b, 0x28) != 0 || buf_u8(b, 0x28) != 0) {
        return -1;
    }
    if (pickle_str(b, "storage") != 0 ||
        pickle_global(b, "torch", "FloatStorage") != 0 || pickle_str(b, key) != 0 ||
        pickle_str(b, "cpu") != 0 || pickle_int(b, t->count) != 0 ||
        buf_u8(b, 0x74) != 0 || buf_u8(b, 0x51) != 0) {
        return -1;
    }
    if (pickle_int(b, 0) != 0 || pickle_tuple(b, t->dims, t->ndim) != 0 ||
        pickle_tuple(b, stride, t->ndim) != 0 || buf_u8(b, 0x89) != 0) {
        return -1;
    }
    if (pickle_global(b, "collections", "OrderedDict") != 0 ||
        buf_u8(b, 0x29) != 0 || buf_u8(b, 0x52) != 0) {
        return -1;
    }
    return buf_u8(b, 0x74) != 0 || buf_u8(b, 0x52);
}

static int build_data_pkl(Buf *b, const PthTensor *t, int n)
{
    int i;
    if (buf_u8(b, 0x80) != 0 || buf_u8(b, 2) != 0 || buf_u8(b, 0x7d) != 0 ||
        buf_u8(b, 0x28) != 0) {
        return -1;
    }
    for (i = 0; i < n; i++) {
        if (pickle_tensor(b, &t[i], i) != 0) {
            return -1;
        }
    }
    return buf_u8(b, 0x75) != 0 || buf_u8(b, 0x2e);
}

static void fill_positional_encoding(int hidden, float *pe)
{
    int pos, i;
    float scale = (float)(-log(10000.0) / (double)hidden);
    for (pos = 0; pos < PE_MAX_LEN; pos++) {
        for (i = 0; i < hidden; i += 2) {
            float div = expf((float)i * scale);
            float arg = (float)pos * div;
            size_t base = (size_t)pos * (size_t)hidden + (size_t)i;
            pe[base] = sinf(arg);
            if (i + 1 < hidden) {
                pe[base + 1] = cosf(arg);
            }
        }
    }
}

static void stem_from_path(const char *path, char *stem, size_t n)
{
    const char *base = path, *p;
    char *dot;
    for (p = path; *p; p++) {
        if (*p == '/' || *p == '\\') {
            base = p + 1;
        }
    }
    snprintf(stem, n, "%s", base);
    dot = strrchr(stem, '.');
    if (dot) {
        *dot = 0;
    }
}

static int pth_path_from_bin(const char *binpath, char *out, size_t n)
{
    size_t len;
    snprintf(out, n, "%s", binpath);
    len = strlen(out);
    if (len >= 4 && strcmp(out + len - 4, ".bin") == 0) {
        memcpy(out + len - 4, ".pth", 4);
        return 0;
    }
    if (len + 4 >= n) {
        return -1;
    }
    memcpy(out + len, ".pth", 5);
    return 0;
}

static void free_tensors(PthTensor *t, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        free(t[i].data);
        t[i].data = NULL;
    }
}

static int read_bin_tensors(const char *path, PthTensor *t, int maxn, int *n_out,
                            int *hidden_out)
{
    FILE *fp = fopen(path, "rb");
    uint32_t magic = 0, ver = 0, nt = 0;
    char code[8] = {0};
    int32_t hdr[6] = {0};
    uint32_t i;
    if (!fp) {
        return ANN_ERR;
    }
    if (fread(&magic, 4, 1, fp) != 1 || magic != ANN_MAGIC ||
        fread(&ver, 4, 1, fp) != 1 || ver != ANN_BIN_VER ||
        fread(code, 8, 1, fp) != 1 || fread(hdr, sizeof(int32_t), 6, fp) != 6 ||
        fread(&nt, 4, 1, fp) != 1) {
        fclose(fp);
        return ANN_ERR;
    }
    if ((int)nt > maxn) {
        fclose(fp);
        return ANN_ERR;
    }
    *hidden_out = hdr[1];
    *n_out = 0;
    for (i = 0; i < nt; i++) {
        int32_t ndim = 0, dims[4] = {0}, nbytes = 0;
        PthTensor *cur = &t[i];
        memset(cur, 0, sizeof(*cur));
        if (fread(cur->name, ANN_NAME_LEN, 1, fp) != 1 ||
            fread(&ndim, 4, 1, fp) != 1 || fread(dims, 4, 4, fp) != 4 ||
            fread(&nbytes, 4, 1, fp) != 1) {
            free_tensors(t, (int)i);
            fclose(fp);
            return ANN_ERR;
        }
        cur->name[ANN_NAME_LEN - 1] = 0;
        cur->ndim = ndim;
        memcpy(cur->dims, dims, sizeof(dims));
        cur->count = nbytes / (int)sizeof(float);
        cur->data = (float *)malloc((size_t)nbytes);
        if (!cur->data || fread(cur->data, 1, (size_t)nbytes, fp) != (size_t)nbytes) {
            free(cur->data);
            free_tensors(t, (int)i);
            fclose(fp);
            return ANN_ERR;
        }
        *n_out = (int)i + 1;
    }
    fclose(fp);
    return ANN_OK;
}

static int write_torch_pth(const char *dst, const PthTensor *t, int n)
{
    FILE *fp;
    ZipRec recs[ZIP_MAX_FILES];
    int nrec = 0, i;
    char stem[64], name[96];
    Buf pkl = {0};
    int rc = ANN_ERR;
    stem_from_path(dst, stem, sizeof(stem));
    if (!stem[0] || build_data_pkl(&pkl, t, n) != 0) {
        free(pkl.p);
        return ANN_ERR;
    }
    fp = fopen(dst, "wb");
    if (!fp) {
        perror(dst);
        free(pkl.p);
        return ANN_ERR;
    }
    snprintf(name, sizeof(name), "%s/data.pkl", stem);
    if (zip_add(fp, recs, &nrec, name, pkl.p, (uint32_t)pkl.n) != 0) {
        goto done;
    }
    snprintf(name, sizeof(name), "%s/version", stem);
    if (zip_add(fp, recs, &nrec, name, "3\n", 2) != 0) {
        goto done;
    }
    snprintf(name, sizeof(name), "%s/byteorder", stem);
    if (zip_add(fp, recs, &nrec, name, "little", 6) != 0) {
        goto done;
    }
    snprintf(name, sizeof(name), "%s/.format_version", stem);
    if (zip_add(fp, recs, &nrec, name, "1", 1) != 0) {
        goto done;
    }
    snprintf(name, sizeof(name), "%s/.storage_alignment", stem);
    if (zip_add(fp, recs, &nrec, name, "64", 2) != 0) {
        goto done;
    }
    for (i = 0; i < n; i++) {
        snprintf(name, sizeof(name), "%s/data/%d", stem, i);
        if (zip_add(fp, recs, &nrec, name, t[i].data,
                    (uint32_t)t[i].count * (uint32_t)sizeof(float)) != 0) {
            goto done;
        }
    }
    if (zip_finish(fp, recs, nrec) == 0) {
        rc = ANN_OK;
    }
done:
    fclose(fp);
    free(pkl.p);
    return rc;
}

static int copy_file_times(const char *src, const char *dst)
{
#ifdef _WIN32
    HANDLE hs, hd;
    FILETIME accessed, written;
    hs = CreateFileA(src, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hs == INVALID_HANDLE_VALUE) {
        return -1;
    }
    if (!GetFileTime(hs, NULL, &accessed, &written)) {
        CloseHandle(hs);
        return -1;
    }
    CloseHandle(hs);
    hd = CreateFileA(dst, FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                     NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hd == INVALID_HANDLE_VALUE) {
        return -1;
    }
    if (!SetFileTime(hd, NULL, &accessed, &written)) {
        CloseHandle(hd);
        return -1;
    }
    CloseHandle(hd);
    return 0;
#else
    struct stat st;
    struct timespec ts[2];
    if (stat(src, &st) != 0) {
        return -1;
    }
    ts[0] = st.st_atim;
    ts[1] = st.st_mtim;
    return utimensat(AT_FDCWD, dst, ts, 0);
#endif
}

int weights_export_pth(const char *binpath)
{
    PthTensor t[PTH_MAX_TENSORS];
    int n = 0, hidden = 0, i, have_pe = 0;
    char dst[512];
    int rc;
    if (!binpath || !binpath[0]) {
        return ANN_ERR;
    }
    memset(t, 0, sizeof(t));
    if (read_bin_tensors(binpath, t, PTH_MAX_TENSORS - 1, &n, &hidden) != ANN_OK) {
        fprintf(stderr, "weights_export_pth: bad bin %s\n", binpath);
        return ANN_ERR;
    }
    for (i = 0; i < n; i++) {
        if (strcmp(t[i].name, "pos_encoder.pe") == 0) {
            have_pe = 1;
            break;
        }
    }
    if (!have_pe && hidden > 0) {
        PthTensor *pe = &t[n];
        memset(pe, 0, sizeof(*pe));
        snprintf(pe->name, ANN_NAME_LEN, "pos_encoder.pe");
        pe->ndim = 3;
        pe->dims[0] = 1;
        pe->dims[1] = PE_MAX_LEN;
        pe->dims[2] = hidden;
        pe->count = PE_MAX_LEN * hidden;
        pe->data = (float *)calloc((size_t)pe->count, sizeof(float));
        if (!pe->data) {
            free_tensors(t, n);
            return ANN_ERR;
        }
        fill_positional_encoding(hidden, pe->data);
        n++;
    }
    if (pth_path_from_bin(binpath, dst, sizeof(dst)) != 0) {
        free_tensors(t, n);
        return ANN_ERR;
    }
    rc = write_torch_pth(dst, t, n);
    free_tensors(t, n);
    if (rc != ANN_OK) {
        fprintf(stderr, "weights_export_pth: write failed %s\n", dst);
        return ANN_ERR;
    }
    (void)copy_file_times(binpath, dst);
    printf("%s -> %s\n", binpath, dst);
    return ANN_OK;
}

#define PK_STACK 256
#define PK_MEMO 512
#define PK_POOL 2048

typedef enum {
    PV_NONE = 0,
    PV_MARK,
    PV_INT,
    PV_STR,
    PV_BOOL,
    PV_GLOBAL,
    PV_TUPLE,
    PV_DICT,
    PV_STORAGE,
    PV_TENSOR,
    PV_OBJ
} PvKind;

typedef struct {
    PvKind kind;
    int64_t i;
    char s[96];
    int n;
    int ch[8];
    int storage_id;
    int count;
    int ndim;
    int dims[4];
} PVal;

typedef struct {
    char name[96];
    uint32_t method, usize, data_off;
} ZipEnt;

typedef struct {
    unsigned char *raw;
    size_t n;
    ZipEnt ent[ZIP_MAX_FILES];
    int nent;
} ZipArc;

typedef struct {
    PthTensor t;
    char key[16];
} ParsedT;

static uint16_t ru16(const unsigned char *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t ru32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static int zip_name_tail(const char *name, const char *tail)
{
    size_t nl = strlen(name), tl = strlen(tail);
    if (nl < tl) {
        return 0;
    }
    if (strcmp(name + nl - tl, tail) != 0) {
        return 0;
    }
    return nl == tl || name[nl - tl - 1] == '/';
}

static int slurp_file(const char *path, unsigned char **out, size_t *n_out)
{
    FILE *fp = fopen(path, "rb");
    unsigned char *p;
    long sz;
    if (!fp) {
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0 || (sz = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    p = (unsigned char *)malloc((size_t)sz);
    if (!p || fread(p, 1, (size_t)sz, fp) != (size_t)sz) {
        free(p);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out = p;
    *n_out = (size_t)sz;
    return 0;
}

static int zip_open_mem(unsigned char *raw, size_t n, ZipArc *z)
{
    size_t i, eocd = (size_t)-1, min_i;
    uint32_t cd_off, cd_size;
    uint16_t nent;
    memset(z, 0, sizeof(*z));
    z->raw = raw;
    z->n = n;
    if (n < 22) {
        return -1;
    }
    min_i = n > (22 + 65535) ? n - (22 + 65535) : 0;
    for (i = n - 22; i >= min_i; i--) {
        if (ru32(raw + i) == 0x06054b50u) {
            eocd = i;
            break;
        }
        if (i == 0) {
            break;
        }
    }
    if (eocd == (size_t)-1) {
        return -1;
    }
    nent = ru16(raw + eocd + 10);
    cd_size = ru32(raw + eocd + 12);
    cd_off = ru32(raw + eocd + 16);
    if ((size_t)cd_off + cd_size > n || nent > ZIP_MAX_FILES) {
        return -1;
    }
    i = cd_off;
    while (z->nent < nent) {
        ZipEnt *e;
        uint16_t nlen, elen, clen, method;
        uint32_t local, usize;
        if (i + 46 > n || ru32(raw + i) != 0x02014b50u) {
            return -1;
        }
        method = ru16(raw + i + 10);
        usize = ru32(raw + i + 24);
        nlen = ru16(raw + i + 28);
        elen = ru16(raw + i + 30);
        clen = ru16(raw + i + 32);
        local = ru32(raw + i + 42);
        if (nlen == 0 || nlen >= sizeof(e->name) || i + 46 + nlen + elen + clen > n) {
            return -1;
        }
        e = &z->ent[z->nent];
        memset(e, 0, sizeof(*e));
        memcpy(e->name, raw + i + 46, nlen);
        e->name[nlen] = 0;
        e->method = method;
        e->usize = usize;
        if (local + 30 + nlen > n || ru32(raw + local) != 0x04034b50u) {
            return -1;
        }
        {
            uint16_t lnlen = ru16(raw + local + 26);
            uint16_t lelen = ru16(raw + local + 28);
            if ((size_t)local + 30 + lnlen + lelen + usize > n) {
                return -1;
            }
            e->data_off = local + 30 + lnlen + lelen;
        }
        z->nent++;
        i += 46 + nlen + elen + clen;
    }
    return 0;
}

static int zip_get(const ZipArc *z, const char *tail, const unsigned char **p, size_t *n)
{
    int i;
    for (i = 0; i < z->nent; i++) {
        if (zip_name_tail(z->ent[i].name, tail)) {
            if (z->ent[i].method != 0) {
                fprintf(stderr, "zip compressed entry %s\n", z->ent[i].name);
                return -1;
            }
            *p = z->raw + z->ent[i].data_off;
            *n = z->ent[i].usize;
            return 0;
        }
    }
    return -1;
}

static int pk_push(PVal *pool, int *np, int *stack, int *nsp, PVal v)
{
    if (*np >= PK_POOL || *nsp >= PK_STACK) {
        return -1;
    }
    pool[*np] = v;
    stack[(*nsp)++] = (*np)++;
    return 0;
}

static int pk_read_nl(const unsigned char *p, size_t n, size_t *i, char *out, size_t cap)
{
    size_t s = *i, len;
    while (*i < n && p[*i] != '\n') {
        (*i)++;
    }
    if (*i >= n) {
        return -1;
    }
    len = *i - s;
    if (len >= cap) {
        len = cap - 1;
    }
    memcpy(out, p + s, len);
    out[len] = 0;
    (*i)++;
    return 0;
}

static int pk_collect_until_mark(PVal *pool, int *stack, int *nsp, int *items, int *nitem,
                                 int maxn)
{
    int tmp[PK_STACK], nt = 0, i;
    while (*nsp > 0 && pool[stack[*nsp - 1]].kind != PV_MARK) {
        if (nt >= PK_STACK) {
            return -1;
        }
        tmp[nt++] = stack[--(*nsp)];
    }
    if (*nsp <= 0) {
        return -1;
    }
    (*nsp)--;
    if (nt > maxn) {
        return -1;
    }
    for (i = 0; i < nt; i++) {
        items[i] = tmp[nt - 1 - i];
    }
    *nitem = nt;
    return 0;
}

static int parse_data_pkl(const unsigned char *p, size_t n, ParsedT *out, int maxn, int *n_out)
{
    PVal pool[PK_POOL];
    int stack[PK_STACK], memo[PK_MEMO];
    int np = 0, nsp = 0, nmemo = 0;
    size_t i = 0;
    int tcount = 0;
    memset(memo, 0xff, sizeof(memo));
    *n_out = 0;
    while (i < n) {
        unsigned char op = p[i++];
        PVal v;
        memset(&v, 0, sizeof(v));
        if (op == 0x2e) { /* STOP */
            break;
        }
        if (op == 0x80) { /* PROTO */
            if (i >= n) {
                return -1;
            }
            i++;
            continue;
        }
        if (op == 0x95) { /* FRAME */
            if (i + 8 > n) {
                return -1;
            }
            i += 8;
            continue;
        }
        if (op == 0x7d) { /* EMPTY_DICT */
            v.kind = PV_DICT;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x28) { /* MARK */
            v.kind = PV_MARK;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x29) { /* EMPTY_TUPLE */
            v.kind = PV_TUPLE;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x4e || op == 0x5d || op == 0x8f) { /* NONE / EMPTY_LIST / EMPTY_SET */
            v.kind = PV_OBJ;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x88 || op == 0x89) { /* NEWTRUE / NEWFALSE */
            v.kind = PV_BOOL;
            v.i = (op == 0x88);
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x4b) { /* BININT1 */
            if (i >= n) {
                return -1;
            }
            v.kind = PV_INT;
            v.i = p[i++];
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x4d) { /* BININT2 */
            if (i + 2 > n) {
                return -1;
            }
            v.kind = PV_INT;
            v.i = ru16(p + i);
            i += 2;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x4a) { /* BININT */
            if (i + 4 > n) {
                return -1;
            }
            v.kind = PV_INT;
            v.i = (int32_t)ru32(p + i);
            i += 4;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x58 || op == 0x8c || op == 0x8d) { /* BINUNICODE family */
            uint64_t len = 0;
            if (op == 0x8c) {
                if (i >= n) {
                    return -1;
                }
                len = p[i++];
            } else if (op == 0x58) {
                if (i + 4 > n) {
                    return -1;
                }
                len = ru32(p + i);
                i += 4;
            } else {
                if (i + 8 > n) {
                    return -1;
                }
                len = (uint64_t)ru32(p + i) | ((uint64_t)ru32(p + i + 4) << 32);
                i += 8;
            }
            if (i + len > n) {
                return -1;
            }
            v.kind = PV_STR;
            if (len >= sizeof(v.s)) {
                memcpy(v.s, p + i, sizeof(v.s) - 1);
                v.s[sizeof(v.s) - 1] = 0;
            } else {
                memcpy(v.s, p + i, (size_t)len);
                v.s[len] = 0;
            }
            i += (size_t)len;
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x63) { /* GLOBAL */
            char mod[96], name[96];
            if (pk_read_nl(p, n, &i, mod, sizeof(mod)) != 0 ||
                pk_read_nl(p, n, &i, name, sizeof(name)) != 0) {
                return -1;
            }
            v.kind = PV_GLOBAL;
            snprintf(v.s, sizeof(v.s), "%s", name);
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x93) { /* STACK_GLOBAL */
            if (nsp < 2) {
                return -1;
            }
            {
                PVal *namev = &pool[stack[--nsp]];
                (void)stack[--nsp];
                v.kind = PV_GLOBAL;
                snprintf(v.s, sizeof(v.s), "%s", namev->s);
            }
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x71 || op == 0x72 || op == 0x94) { /* BINPUT / LONG_BINPUT / MEMOIZE */
            unsigned int idx;
            if (nsp <= 0) {
                return -1;
            }
            if (op == 0x71) {
                if (i >= n) {
                    return -1;
                }
                idx = p[i++];
            } else if (op == 0x72) {
                if (i + 4 > n) {
                    return -1;
                }
                idx = ru32(p + i);
                i += 4;
            } else {
                idx = (unsigned int)nmemo++;
            }
            if (idx >= PK_MEMO) {
                return -1;
            }
            memo[idx] = stack[nsp - 1];
            if ((int)idx + 1 > nmemo) {
                nmemo = (int)idx + 1;
            }
            continue;
        }
        if (op == 0x68 || op == 0x6a) { /* BINGET / LONG_BINGET */
            unsigned int idx;
            if (op == 0x68) {
                if (i >= n) {
                    return -1;
                }
                idx = p[i++];
            } else {
                if (i + 4 > n) {
                    return -1;
                }
                idx = ru32(p + i);
                i += 4;
            }
            if (idx >= PK_MEMO || memo[idx] < 0) {
                return -1;
            }
            if (nsp >= PK_STACK) {
                return -1;
            }
            stack[nsp++] = memo[idx];
            continue;
        }
        if (op == 0x85 || op == 0x86 || op == 0x87) { /* TUPLE1/2/3 */
            int want = op - 0x84, k;
            if (nsp < want) {
                return -1;
            }
            v.kind = PV_TUPLE;
            v.n = want;
            for (k = 0; k < want; k++) {
                v.ch[want - 1 - k] = stack[--nsp];
            }
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x74) { /* TUPLE */
            int items[8], ni = 0, k;
            if (pk_collect_until_mark(pool, stack, &nsp, items, &ni, 8) != 0) {
                return -1;
            }
            v.kind = PV_TUPLE;
            v.n = ni;
            for (k = 0; k < ni; k++) {
                v.ch[k] = items[k];
            }
            if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                return -1;
            }
            continue;
        }
        if (op == 0x51) { /* BINPERSID */
            PVal *top, st;
            if (nsp <= 0) {
                return -1;
            }
            top = &pool[stack[nsp - 1]];
            if (top->kind != PV_TUPLE || top->n < 5) {
                return -1;
            }
            memset(&st, 0, sizeof(st));
            st.kind = PV_STORAGE;
            {
                PVal *key = &pool[top->ch[2]];
                PVal *cnt = &pool[top->ch[4]];
                if (key->kind == PV_STR) {
                    snprintf(st.s, sizeof(st.s), "%s", key->s);
                    st.storage_id = atoi(key->s);
                }
                if (cnt->kind == PV_INT) {
                    st.count = (int)cnt->i;
                }
            }
            if (np >= PK_POOL) {
                return -1;
            }
            pool[np] = st;
            stack[nsp - 1] = np++;
            continue;
        }
        if (op == 0x52 || op == 0x81) { /* REDUCE / NEWOBJ */
            PVal *args, *fn;
            if (nsp < 2) {
                return -1;
            }
            args = &pool[stack[--nsp]];
            fn = &pool[stack[--nsp]];
            if (fn->kind == PV_GLOBAL && strcmp(fn->s, "_rebuild_tensor_v2") == 0 &&
                args->kind == PV_TUPLE && args->n >= 3) {
                PVal *stor = &pool[args->ch[0]];
                PVal *shape = &pool[args->ch[2]];
                int k;
                memset(&v, 0, sizeof(v));
                v.kind = PV_TENSOR;
                if (stor->kind == PV_STORAGE) {
                    snprintf(v.s, sizeof(v.s), "%s", stor->s);
                    v.storage_id = stor->storage_id;
                    v.count = stor->count;
                }
                if (shape->kind == PV_TUPLE) {
                    v.ndim = shape->n > 4 ? 4 : shape->n;
                    for (k = 0; k < v.ndim; k++) {
                        v.dims[k] = (int)pool[shape->ch[k]].i;
                    }
                }
                if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                    return -1;
                }
            } else {
                v.kind = PV_OBJ;
                if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                    return -1;
                }
            }
            continue;
        }
        if (op == 0x75 || op == 0x73) { /* SETITEMS / SETITEM */
            int items[PK_STACK], ni = 0, k;
            if (op == 0x73) {
                if (nsp < 3) {
                    return -1;
                }
                items[1] = stack[--nsp];
                items[0] = stack[--nsp];
                ni = 2;
            } else if (pk_collect_until_mark(pool, stack, &nsp, items, &ni, PK_STACK) != 0) {
                return -1;
            }
            if (ni % 2 != 0) {
                return -1;
            }
            for (k = 0; k < ni; k += 2) {
                PVal *key = &pool[items[k]];
                PVal *val = &pool[items[k + 1]];
                const char *nm;
                if (key->kind != PV_STR || val->kind != PV_TENSOR) {
                    continue;
                }
                nm = key->s;
                if (strncmp(nm, "module.", 7) == 0) {
                    nm += 7;
                }
                if (strcmp(nm, "pos_encoder.pe") == 0) {
                    continue;
                }
                if (tcount >= maxn) {
                    return -1;
                }
                memset(&out[tcount], 0, sizeof(out[tcount]));
                snprintf(out[tcount].t.name, ANN_NAME_LEN, "%.*s", ANN_NAME_LEN - 1, nm);
                snprintf(out[tcount].key, sizeof(out[tcount].key), "%s", val->s);
                out[tcount].t.ndim = val->ndim;
                memcpy(out[tcount].t.dims, val->dims, sizeof(val->dims));
                out[tcount].t.count = val->count;
                tcount++;
            }
            if (nsp <= 0 || pool[stack[nsp - 1]].kind != PV_DICT) {
                v.kind = PV_DICT;
                if (pk_push(pool, &np, stack, &nsp, v) != 0) {
                    return -1;
                }
            }
            continue;
        }
        if (op == 0x62) { /* BUILD: pop state, keep instance */
            if (nsp < 2) {
                return -1;
            }
            nsp--;
            continue;
        }
        fprintf(stderr, "weights_import_pth: unsupported pickle op 0x%02x at %zu\n", op, i - 1);
        return -1;
    }
    *n_out = tcount;
    return tcount > 0 ? 0 : -1;
}

static int layer_index(const char *name)
{
    const char *p = strstr(name, "transformer.layers.");
    if (!p) {
        return -1;
    }
    p += strlen("transformer.layers.");
    if (*p < '0' || *p > '9') {
        return -1;
    }
    return atoi(p);
}

static void infer_arch(const ParsedT *t, int n, int *insize, int *hidden, int *outsize,
                       int *nlayers, int *nhead, int *d_ff)
{
    int i, li, max_layer = -1;
    *insize = ANN_INSIZE;
    *hidden = ANN_HIDDEN;
    *outsize = ANN_OUTSIZE;
    *nlayers = ANN_NLAYERS;
    *nhead = ANN_NHEAD;
    *d_ff = ANN_D_FF;
    for (i = 0; i < n; i++) {
        if (strcmp(t[i].t.name, "embedding.weight") == 0 && t[i].t.ndim == 2) {
            *hidden = t[i].t.dims[0];
            *insize = t[i].t.dims[1];
        } else if (strcmp(t[i].t.name, "fc.weight") == 0 && t[i].t.ndim == 2) {
            *outsize = t[i].t.dims[0];
        } else if (strstr(t[i].t.name, "linear1.weight") && t[i].t.ndim == 2) {
            *d_ff = t[i].t.dims[0];
        }
        li = layer_index(t[i].t.name);
        if (li > max_layer) {
            max_layer = li;
        }
    }
    if (max_layer >= 0) {
        *nlayers = max_layer + 1;
    }
}

static int write_bin_from_parsed(const char *path, const char *code, const ParsedT *t, int n)
{
    FILE *fp;
    uint32_t magic = ANN_MAGIC, ver = ANN_BIN_VER, nt = (uint32_t)n;
    int32_t hdr[6];
    int insize, hidden, outsize, nlayers, nhead, d_ff, i;
    char code8[8] = {0};
    infer_arch(t, n, &insize, &hidden, &outsize, &nlayers, &nhead, &d_ff);
    hdr[0] = insize;
    hdr[1] = hidden;
    hdr[2] = outsize;
    hdr[3] = nlayers;
    hdr[4] = nhead;
    hdr[5] = d_ff;
    snprintf(code8, sizeof(code8), "%.6s", code);
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        return ANN_ERR;
    }
    fwrite(&magic, 4, 1, fp);
    fwrite(&ver, 4, 1, fp);
    fwrite(code8, 8, 1, fp);
    fwrite(hdr, sizeof(int32_t), 6, fp);
    fwrite(&nt, 4, 1, fp);
    for (i = 0; i < n; i++) {
        int32_t ndim = t[i].t.ndim;
        int32_t dims[4] = {0};
        int32_t nbytes = t[i].t.count * (int32_t)sizeof(float);
        char name[ANN_NAME_LEN];
        memcpy(name, t[i].t.name, ANN_NAME_LEN);
        name[ANN_NAME_LEN - 1] = 0;
        memcpy(dims, t[i].t.dims, sizeof(int) * 4);
        fwrite(name, ANN_NAME_LEN, 1, fp);
        fwrite(&ndim, 4, 1, fp);
        fwrite(dims, 4, 4, fp);
        fwrite(&nbytes, 4, 1, fp);
        if (fwrite(t[i].t.data, 1, (size_t)nbytes, fp) != (size_t)nbytes) {
            fclose(fp);
            return ANN_ERR;
        }
    }
    fclose(fp);
    return ANN_OK;
}

static int bin_path_from_pth(const char *pthpath, char *out, size_t n)
{
    size_t len;
    snprintf(out, n, "%s", pthpath);
    len = strlen(out);
    if (len >= 4 && strcmp(out + len - 4, ".pth") == 0) {
        memcpy(out + len - 4, ".bin", 4);
        return 0;
    }
    if (len + 4 >= n) {
        return -1;
    }
    memcpy(out + len, ".bin", 5);
    return 0;
}

static int file_mtime_ns(const char *path, int64_t *out)
{
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path, &st) != 0) {
        return -1;
    }
    *out = (int64_t)st.st_mtime * 1000000000LL;
    return 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    *out = (int64_t)st.st_mtim.tv_sec * 1000000000LL + (int64_t)st.st_mtim.tv_nsec;
    return 0;
#endif
}

int weights_import_pth(const char *pthpath)
{
    unsigned char *raw = NULL;
    size_t rawn = 0, pkl_n = 0, blob_n = 0;
    const unsigned char *pkl = NULL, *blob = NULL;
    ZipArc z;
    ParsedT parsed[PTH_MAX_TENSORS];
    int n = 0, i, rc = ANN_ERR;
    char dst[512], stem[64], tail[48];
    if (!pthpath || !pthpath[0]) {
        return ANN_ERR;
    }
    if (slurp_file(pthpath, &raw, &rawn) != 0 || zip_open_mem(raw, rawn, &z) != 0) {
        fprintf(stderr, "weights_import_pth: not a zip pth %s\n", pthpath);
        free(raw);
        return ANN_ERR;
    }
    if (zip_get(&z, "data.pkl", &pkl, &pkl_n) != 0 ||
        parse_data_pkl(pkl, pkl_n, parsed, PTH_MAX_TENSORS, &n) != 0) {
        fprintf(stderr, "weights_import_pth: pickle failed %s\n", pthpath);
        free(raw);
        return ANN_ERR;
    }
    for (i = 0; i < n; i++) {
        snprintf(tail, sizeof(tail), "data/%s", parsed[i].key[0] ? parsed[i].key : "0");
        if (zip_get(&z, tail, &blob, &blob_n) != 0) {
            fprintf(stderr, "weights_import_pth: missing storage %s\n", tail);
            goto done;
        }
        if (blob_n != (size_t)parsed[i].t.count * sizeof(float)) {
            fprintf(stderr, "weights_import_pth: size mismatch %s\n", parsed[i].t.name);
            goto done;
        }
        parsed[i].t.data = (float *)malloc(blob_n);
        if (!parsed[i].t.data) {
            goto done;
        }
        memcpy(parsed[i].t.data, blob, blob_n);
    }
    if (bin_path_from_pth(pthpath, dst, sizeof(dst)) != 0) {
        goto done;
    }
    stem_from_path(pthpath, stem, sizeof(stem));
    if (write_bin_from_parsed(dst, stem, parsed, n) != ANN_OK) {
        goto done;
    }
    (void)copy_file_times(pthpath, dst);
    printf("%s -> %s\n", pthpath, dst);
    rc = ANN_OK;
done:
    for (i = 0; i < n; i++) {
        free(parsed[i].t.data);
        parsed[i].t.data = NULL;
    }
    free(raw);
    return rc;
}

int weights_sync_bin(const char *code)
{
    char binpath[256], pthpath[256];
    int64_t pth_t = 0, bin_t = 0;
    if (!code || !code[0]) {
        return ANN_ERR;
    }
    snprintf(binpath, sizeof(binpath), "./ann/%s.bin", code);
    snprintf(pthpath, sizeof(pthpath), "./ann/%s.pth", code);
    if (file_mtime_ns(pthpath, &pth_t) != 0) {
        return ANN_OK;
    }
    if (file_mtime_ns(binpath, &bin_t) == 0 && pth_t <= bin_t) {
        return ANN_OK;
    }
    printf("pth newer than bin, converting %s\n", pthpath);
    return weights_import_pth(pthpath);
}
