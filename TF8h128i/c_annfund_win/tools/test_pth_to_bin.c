#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#define mkdir_p(p) _mkdir(p)
#define chdir_p(p) _chdir(p)
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#define mkdir_p(p) mkdir((p), 0755)
#define chdir_p(p) chdir(p)
#endif

#define MAGIC 0x424E4E41u
#define NAME_LEN 64

static void wr_name(FILE *fp, const char *s)
{
    char buf[NAME_LEN];
    memset(buf, 0, sizeof(buf));
    snprintf(buf, sizeof(buf), "%s", s);
    fwrite(buf, NAME_LEN, 1, fp);
}

static int write_tiny_bin(const char *path)
{
    FILE *fp = fopen(path, "wb");
    uint32_t magic = MAGIC, ver = 1, nt = 2;
    int32_t hdr[6] = {4, 2, 2, 1, 1, 8};
    char code[8] = {0};
    float w[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[2] = {9, 10};
    int32_t ndim, dims[4], nbytes;
    memcpy(code, "000000", 6);
    if (!fp) {
        perror(path);
        return -1;
    }
    fwrite(&magic, 4, 1, fp);
    fwrite(&ver, 4, 1, fp);
    fwrite(code, 8, 1, fp);
    fwrite(hdr, 4, 6, fp);
    fwrite(&nt, 4, 1, fp);

    wr_name(fp, "embedding.weight");
    ndim = 2;
    dims[0] = 2;
    dims[1] = 4;
    dims[2] = 0;
    dims[3] = 0;
    nbytes = 32;
    fwrite(&ndim, 4, 1, fp);
    fwrite(dims, 4, 4, fp);
    fwrite(&nbytes, 4, 1, fp);
    fwrite(w, 1, 32, fp);

    wr_name(fp, "embedding.bias");
    ndim = 1;
    dims[0] = 2;
    dims[1] = 0;
    dims[2] = 0;
    dims[3] = 0;
    nbytes = 8;
    fwrite(&ndim, 4, 1, fp);
    fwrite(dims, 4, 4, fp);
    fwrite(&nbytes, 4, 1, fp);
    fwrite(b, 1, 8, fp);
    fclose(fp);
    return 0;
}

static int find_tensor(FILE *fp, const char *want, float *out, int nwant)
{
    uint32_t magic = 0, ver = 0, nt = 0;
    char code[8];
    int32_t hdr[6];
    uint32_t i;
    if (fread(&magic, 4, 1, fp) != 1 || magic != MAGIC ||
        fread(&ver, 4, 1, fp) != 1 || fread(code, 8, 1, fp) != 1 ||
        fread(hdr, 4, 6, fp) != 6 || fread(&nt, 4, 1, fp) != 1) {
        return -1;
    }
    for (i = 0; i < nt; i++) {
        char name[NAME_LEN];
        int32_t ndim = 0, dims[4] = {0}, nbytes = 0;
        if (fread(name, NAME_LEN, 1, fp) != 1 || fread(&ndim, 4, 1, fp) != 1 ||
            fread(dims, 4, 4, fp) != 4 || fread(&nbytes, 4, 1, fp) != 1) {
            return -1;
        }
        name[NAME_LEN - 1] = 0;
        if (strcmp(name, want) == 0) {
            if (nbytes != nwant * (int)sizeof(float)) {
                return -1;
            }
            return fread(out, 1, (size_t)nbytes, fp) == (size_t)nbytes ? 0 : -1;
        }
        if (fseek(fp, nbytes, SEEK_CUR) != 0) {
            return -1;
        }
    }
    return -1;
}

static int expect_floats(const char *path, const char *name, const float *exp, int n)
{
    FILE *fp = fopen(path, "rb");
    float got[16];
    int i;
    if (!fp) {
        fprintf(stderr, "missing %s\n", path);
        return -1;
    }
    if (find_tensor(fp, name, got, n) != 0) {
        fclose(fp);
        fprintf(stderr, "tensor %s not found in %s\n", name, path);
        return -1;
    }
    fclose(fp);
    for (i = 0; i < n; i++) {
        if (got[i] != exp[i]) {
            fprintf(stderr, "%s[%d] got %g want %g\n", name, i, got[i], exp[i]);
            return -1;
        }
    }
    return 0;
}

static int check_tiny(const char *bin)
{
    const float w[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b[2] = {9, 10};
    return expect_floats(bin, "embedding.weight", w, 8) == 0 &&
                   expect_floats(bin, "embedding.bias", b, 2) == 0
               ? 0
               : -1;
}

static int test_sync(const char *dir)
{
    char cwd[512], anndir[300], bin[300], pth[300];
    FILE *fp;
    char marker[4] = {'O', 'L', 'D', 0};
#ifdef _WIN32
    if (!_getcwd(cwd, (int)sizeof(cwd))) {
        return -1;
    }
#else
    if (!getcwd(cwd, sizeof(cwd))) {
        return -1;
    }
#endif
    snprintf(anndir, sizeof(anndir), "%s/ann", dir);
    mkdir_p(anndir);
    if (chdir_p(dir) != 0) {
        perror(dir);
        return -1;
    }
    snprintf(bin, sizeof(bin), "./ann/000001.bin");
    snprintf(pth, sizeof(pth), "./ann/000001.pth");
    if (write_tiny_bin(bin) != 0 || weights_export_pth(bin) != ANN_OK) {
        chdir_p(cwd);
        return -1;
    }
    /* bin newer than pth: skip */
    fp = fopen(bin, "ab");
    if (!fp) {
        chdir_p(cwd);
        return -1;
    }
    fwrite(marker, 1, 3, fp);
    fclose(fp);
    if (weights_sync_bin("000001") != ANN_OK) {
        chdir_p(cwd);
        fprintf(stderr, "sync skip path failed\n");
        return -1;
    }
    fp = fopen(bin, "rb");
    if (!fp) {
        chdir_p(cwd);
        return -1;
    }
    fseek(fp, -3, SEEK_END);
    if (fread(marker, 1, 3, fp) != 3 || marker[0] != 'O' || marker[1] != 'L' ||
        marker[2] != 'D') {
        fclose(fp);
        chdir_p(cwd);
        fprintf(stderr, "sync overwrote bin that was newer\n");
        return -1;
    }
    fclose(fp);

    /* pth newer than bin: convert */
    fp = fopen(bin, "wb");
    fwrite("STALE", 1, 5, fp);
    fclose(fp);
#ifdef _WIN32
    Sleep(1100);
    {
        HANDLE h = CreateFileA(pth, FILE_WRITE_ATTRIBUTES,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, NULL);
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        if (h == INVALID_HANDLE_VALUE || !SetFileTime(h, NULL, NULL, &ft)) {
            if (h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
            }
            chdir_p(cwd);
            fprintf(stderr, "touch pth failed\n");
            return -1;
        }
        CloseHandle(h);
    }
#else
    sleep(1);
    if (utimensat(AT_FDCWD, pth, NULL, 0) != 0) {
        chdir_p(cwd);
        perror(pth);
        return -1;
    }
#endif
    if (weights_sync_bin("000001") != ANN_OK) {
        chdir_p(cwd);
        fprintf(stderr, "sync convert path failed\n");
        return -1;
    }
    chdir_p(cwd);
    snprintf(bin, sizeof(bin), "%s/ann/000001.bin", dir);
    if (check_tiny(bin) != 0) {
        fprintf(stderr, "sync did not refresh bin from newer pth\n");
        return -1;
    }
    printf("pth/bin mtime sync ok\n");
    return 0;
}

int main(int argc, char **argv)
{
    char dir[256], bin[300], pth[300];
    if (argc >= 2) {
        if (weights_import_pth(argv[1]) != ANN_OK) {
            return 1;
        }
        return 0;
    }
#ifdef _WIN32
    snprintf(dir, sizeof(dir), "%s\\ann_pth2bin_XXXXXX", getenv("TEMP") ? getenv("TEMP") : ".");
    if (!_mktemp(dir)) {
        fprintf(stderr, "temp dir failed\n");
        return 1;
    }
#else
    snprintf(dir, sizeof(dir), "/tmp/ann_pth2bin_XXXXXX");
    if (!mkdtemp(dir)) {
        perror("mkdtemp");
        return 1;
    }
#endif
#ifdef _WIN32
    if (mkdir_p(dir) != 0) {
        perror(dir);
        return 1;
    }
#endif
    snprintf(bin, sizeof(bin), "%s/000000.bin", dir);
    snprintf(pth, sizeof(pth), "%s/000000.pth", dir);
    if (write_tiny_bin(bin) != 0) {
        return 1;
    }
    if (weights_export_pth(bin) != ANN_OK) {
        fprintf(stderr, "export pth failed\n");
        return 1;
    }
    remove(bin);
    if (weights_import_pth(pth) != ANN_OK) {
        fprintf(stderr, "import pth failed\n");
        return 1;
    }
    if (check_tiny(bin) != 0) {
        return 1;
    }
    printf("pth->bin roundtrip ok\n");
    if (test_sync(dir) != 0) {
        return 1;
    }
    return 0;
}
