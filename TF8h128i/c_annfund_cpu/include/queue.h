#ifndef C_ANNFUND_QUEUE_H
#define C_ANNFUND_QUEUE_H

#include "ann.h"
#include <pthread.h>

typedef struct {
    char codes[ANN_MAX_CODES][8];
    int n;
    pthread_mutex_t mu;
} CodeList;

void clist_init(CodeList *l);
void clist_lock(CodeList *l);
void clist_unlock(CodeList *l);
int clist_push_back(CodeList *l, const char *code);
int clist_push_front(CodeList *l, const char *code);
int clist_pop_front(CodeList *l, char *out);
int clist_remove(CodeList *l, const char *code);
int clist_contains(const CodeList *l, const char *code);
void clist_reverse(CodeList *l);
void clist_sort(CodeList *l, int descending);
void clist_shuffle(CodeList *l, uint32_t *rng);
int clist_index(const CodeList *l, const char *code);
void clist_extend_unique(CodeList *l, const CodeList *src, const CodeList *exclude_a, const CodeList *exclude_b);
void clist_subtract(CodeList *l, const CodeList *rm);

int scan_ann_pool(const char *dir, CodeList *pool);
int pool_group_index(int ind, int total, int parts);
void slice_pool(const CodeList *pool, int task_count, int this_task, CodeList *out);

#endif
