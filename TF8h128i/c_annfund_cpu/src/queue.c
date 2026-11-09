#include "queue.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

void clist_init(CodeList *l)
{
    memset(l, 0, sizeof(*l));
    pthread_mutex_init(&l->mu, NULL);
}

void clist_lock(CodeList *l) { pthread_mutex_lock(&l->mu); }
void clist_unlock(CodeList *l) { pthread_mutex_unlock(&l->mu); }

static int valid_code(const char *c)
{
    if (!c || strlen(c) != 6) {
        return 0;
    }
    for (int i = 0; i < 6; i++) {
        if (c[i] < '0' || c[i] > '9') {
            return 0;
        }
    }
    return 1;
}

int clist_contains(const CodeList *l, const char *code)
{
    for (int i = 0; i < l->n; i++) {
        if (strcmp(l->codes[i], code) == 0) {
            return 1;
        }
    }
    return 0;
}

int clist_index(const CodeList *l, const char *code)
{
    for (int i = 0; i < l->n; i++) {
        if (strcmp(l->codes[i], code) == 0) {
            return i;
        }
    }
    return -1;
}

int clist_push_back(CodeList *l, const char *code)
{
    if (!valid_code(code) || l->n >= ANN_MAX_CODES) {
        return ANN_ERR;
    }
    memcpy(l->codes[l->n], code, 7);
    l->n++;
    return ANN_OK;
}

int clist_push_front(CodeList *l, const char *code)
{
    if (!valid_code(code) || l->n >= ANN_MAX_CODES) {
        return ANN_ERR;
    }
    memmove(l->codes[1], l->codes[0], (size_t)l->n * sizeof(l->codes[0]));
    memcpy(l->codes[0], code, 7);
    l->n++;
    return ANN_OK;
}

int clist_pop_front(CodeList *l, char *out)
{
    if (l->n <= 0) {
        return ANN_ERR;
    }
    memcpy(out, l->codes[0], 8);
    memmove(l->codes[0], l->codes[1], (size_t)(l->n - 1) * sizeof(l->codes[0]));
    l->n--;
    return ANN_OK;
}

int clist_remove(CodeList *l, const char *code)
{
    int i = clist_index(l, code);
    if (i < 0) {
        return ANN_ERR;
    }
    memmove(l->codes[i], l->codes[i + 1], (size_t)(l->n - i - 1) * sizeof(l->codes[0]));
    l->n--;
    return ANN_OK;
}

void clist_reverse(CodeList *l)
{
    for (int i = 0, j = l->n - 1; i < j; i++, j--) {
        char t[8];
        memcpy(t, l->codes[i], 8);
        memcpy(l->codes[i], l->codes[j], 8);
        memcpy(l->codes[j], t, 8);
    }
}

static int cmp_asc(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }
static int cmp_desc(const void *a, const void *b) { return strcmp((const char *)b, (const char *)a); }

void clist_sort(CodeList *l, int descending)
{
    qsort(l->codes, (size_t)l->n, sizeof(l->codes[0]), descending ? cmp_desc : cmp_asc);
}

void clist_shuffle(CodeList *l, uint32_t *rng)
{
    uint32_t s = *rng ? *rng : 1u;
    for (int i = l->n - 1; i > 0; i--) {
        s = s * 1664525u + 1013904223u;
        int j = (int)(s % (uint32_t)(i + 1));
        char t[8];
        memcpy(t, l->codes[i], 8);
        memcpy(l->codes[i], l->codes[j], 8);
        memcpy(l->codes[j], t, 8);
    }
    *rng = s;
}

void clist_extend_unique(CodeList *l, const CodeList *src, const CodeList *exclude_a, const CodeList *exclude_b)
{
    for (int i = 0; i < src->n; i++) {
        const char *c = src->codes[i];
        if (clist_contains(l, c)) {
            continue;
        }
        if (exclude_a && clist_contains(exclude_a, c)) {
            continue;
        }
        if (exclude_b && clist_contains(exclude_b, c)) {
            continue;
        }
        clist_push_back(l, c);
    }
}

void clist_subtract(CodeList *l, const CodeList *rm)
{
    for (int i = 0; i < rm->n; i++) {
        clist_remove(l, rm->codes[i]);
    }
}

int scan_ann_pool(const char *dir, CodeList *pool)
{
    pool->n = 0;
    DIR *d = opendir(dir);
    if (!d) {
        perror(dir);
        return ANN_ERR;
    }
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        const char *name = e->d_name;
        size_t len = strlen(name);
        if (len == 10 && memcmp(name + 6, ".pth", 4) == 0) {
            char code[8] = {0};
            memcpy(code, name, 6);
            if (valid_code(code) && !clist_contains(pool, code)) {
                clist_push_back(pool, code);
            }
        }
    }
    closedir(d);
    clist_sort(pool, 0);
    return ANN_OK;
}

int pool_group_index(int ind, int total, int parts)
{
    if (ind < 0 || parts <= 0 || total <= 0) {
        return 0;
    }
    int count_per = total / parts;
    if (count_per == 0) {
        return 0;
    }
    if (ind >= (parts - 1) * count_per) {
        return parts;
    }
    return ind / count_per + 1;
}

void slice_pool(const CodeList *pool, int task_count, int this_task, CodeList *out)
{
    out->n = 0;
    if (task_count <= 0 || this_task <= 0 || pool->n < task_count) {
        return;
    }
    int per = pool->n / task_count;
    int start = (this_task - 1) * per;
    int end = (this_task < task_count) ? this_task * per : pool->n;
    for (int i = start; i < end; i++) {
        clist_push_back(out, pool->codes[i]);
    }
}
