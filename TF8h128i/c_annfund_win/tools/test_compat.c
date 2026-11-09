#include "queue.h"
#include "win_compat.h"

#include <stdio.h>
#include <string.h>

static int write_empty(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    fclose(f);
    return 0;
}

int main(void)
{
    ann_mutex_t mu;
    ann_mutex_init(&mu);
    if (ann_mutex_trylock(&mu) != 0) {
        fprintf(stderr, "trylock failed\n");
        return 1;
    }
    if (ann_mutex_trylock(&mu) == 0) {
        fprintf(stderr, "trylock should be busy\n");
        return 2;
    }
    ann_mutex_unlock(&mu);
    ann_mutex_destroy(&mu);

    if (ann_mkdir("c_annfund_win/tools/_scan_tmp") != 0) {
        fprintf(stderr, "mkdir failed\n");
        return 3;
    }
    if (write_empty("c_annfund_win/tools/_scan_tmp/001245.bin") != 0 ||
        write_empty("c_annfund_win/tools/_scan_tmp/000001.pth") != 0 ||
        write_empty("c_annfund_win/tools/_scan_tmp/notacode.txt") != 0) {
        fprintf(stderr, "write fixtures failed\n");
        return 4;
    }

    CodeList pool;
    clist_init(&pool);
    if (scan_ann_pool("c_annfund_win/tools/_scan_tmp", &pool) != ANN_OK) {
        fprintf(stderr, "scan_ann_pool failed\n");
        return 5;
    }
    if (pool.n != 1 || strcmp(pool.codes[0], "000001") != 0) {
        fprintf(stderr, "scan mismatch n=%d code=%s\n", pool.n,
                pool.n > 0 ? pool.codes[0] : "");
        return 6;
    }
    printf("test_compat ok\n");
    return 0;
}
