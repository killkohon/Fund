#define _POSIX_C_SOURCE 200809L

#include "ann.h"
#include "db.h"
#include "dataset.h"
#include "model.h"
#include "queue.h"
#include "weights.h"

#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define LOSS_THRESH 0.0003f
#define DEFAULT_LR 1e-6f
#define DEFAULT_EPOCH 20001

static CodeList codepool;
static CodeList fundcodes;
static CodeList processing;
static CodeList inflight;

static pthread_mutex_t state_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t train_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t run_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t state_cv = PTHREAD_COND_INITIALIZER;
#define ANN_MAX_RUNNING 32
static Model *g_running[ANN_MAX_RUNNING];

static void running_register(Model *m)
{
    pthread_mutex_lock(&run_mu);
    for (int i = 0; i < ANN_MAX_RUNNING; i++) {
        if (g_running[i] == NULL) {
            g_running[i] = m;
            break;
        }
    }
    pthread_mutex_unlock(&run_mu);
}

static void running_unregister(Model *m)
{
    pthread_mutex_lock(&run_mu);
    for (int i = 0; i < ANN_MAX_RUNNING; i++) {
        if (g_running[i] == m) {
            g_running[i] = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&run_mu);
}

static volatile int Quit_flag = 0;
static volatile int ThreadReduce = 0;
static volatile int thread_count = 0;
static volatile int train_epoch = DEFAULT_EPOCH;
static int taskCount = 6;
static int thisTask = 1;
static float train_lr = DEFAULT_LR;

static int cpu_parallel_enabled(void)
{
    const char *e = getenv("ANN_CPU_PARALLEL");
    return e && e[0] == '1' && e[1] == '\0';
}

static int path_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static void ensure_ann_dir(void) { mkdir("./ann", 0755); }

static int train_one(const char *code, int epoch, float lr)
{
    NavSeries nav;
    Dataset ds;
    Model m;
    char binpath[256];
    memset(&nav, 0, sizeof(nav));
    memset(&ds, 0, sizeof(ds));
    memset(&m, 0, sizeof(m));

    weights_bin_path(code, binpath, sizeof(binpath));
    if (weights_sync_bin(code) != ANN_OK) {
        fprintf(stderr, "pth->bin sync failed for %s\n", code);
    }

    int parallel = cpu_parallel_enabled();
    if (!parallel) {
        if (pthread_mutex_trylock(&train_mu) != 0) {
            printf("%s waiting for train slot\n", code);
            pthread_mutex_lock(&train_mu);
        }
    }
    if (model_create(&m, code, 1) != ANN_OK) {
        if (!parallel) {
            pthread_mutex_unlock(&train_mu);
        }
        fprintf(stderr, "model_create failed %s\n", code);
        return ANN_ERR;
    }
    if (path_exists(binpath)) {
        if (weights_load(&m, binpath) != ANN_OK) {
            fprintf(stderr, "load %s failed, using xavier init\n", binpath);
        }
    }

    if (db_load_nav(code, ANN_NAV_TRAIN, &nav) != ANN_OK) {
        fprintf(stderr, "no nav for %s\n", code);
        model_destroy(&m);
        if (!parallel) {
            pthread_mutex_unlock(&train_mu);
        }
        return ANN_ERR;
    }
    if (dataset_from_serial_train(&nav, &ds) != ANN_OK) {
        nav_free(&nav);
        model_destroy(&m);
        if (!parallel) {
            pthread_mutex_unlock(&train_mu);
        }
        return ANN_ERR;
    }

    time_t t0 = time(NULL);
    running_register(&m);
    int rc = model_train(&m, &ds, epoch, lr, LOSS_THRESH, 1);
    running_unregister(&m);
    if (rc != ANN_OK) {
        if (cpu_harden_reset("train") != ANN_OK) {
            memset(&m, 0, sizeof(m));
        } else {
            model_destroy(&m);
        }
    } else {
        weights_save(&m, binpath);
        model_destroy(&m);
    }
    time_t t1 = time(NULL);
    printf("\033[42m\033[37m%s duration:%ld\033[0m\n", code, (long)(t1 - t0));
    dataset_free(&ds);
    nav_free(&nav);
    if (!parallel) {
        pthread_mutex_unlock(&train_mu);
    }
    return rc;
}

static void *train_fund(void *arg)
{
    int id = (int)(intptr_t)arg;
    int epoch = train_epoch;
    float lr = train_lr;
    while (!Quit_flag) {
        char code[8] = {0};
        pthread_mutex_lock(&state_mu);
        while (inflight.n <= 0 && fundcodes.n <= 0 && !Quit_flag && ThreadReduce <= 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            pthread_cond_timedwait(&state_cv, &state_mu, &ts);
        }
        if (Quit_flag) {
            pthread_mutex_unlock(&state_mu);
            break;
        }
        if (ThreadReduce > 0) {
            ThreadReduce--;
            thread_count--;
            printf("Reduce thread:%d\n", thread_count);
            pthread_mutex_unlock(&state_mu);
            break;
        }
        if (inflight.n <= 0 && fundcodes.n > 0) {
            clist_pop_front(&fundcodes, code);
        } else if (clist_pop_front(&inflight, code) != ANN_OK) {
            pthread_mutex_unlock(&state_mu);
            continue;
        }
        printf("take %s, %d tasks left in queue.\n", code, fundcodes.n);
        clist_push_back(&processing, code);
        pthread_mutex_unlock(&state_mu);

        if (train_epoch > 0) {
            epoch = train_epoch;
        }
        if (train_one(code, epoch, lr) != ANN_OK && !Quit_flag) {
            fprintf(stderr, "thread[%d] error training %s\n", id, code);
        }

        pthread_mutex_lock(&state_mu);
        clist_remove(&processing, code);
        if (ThreadReduce > 0) {
            ThreadReduce--;
            thread_count--;
            printf("Reduce thread:%d\n", thread_count);
            pthread_mutex_unlock(&state_mu);
            break;
        }
        if (fundcodes.n > 0 && !Quit_flag) {
            char next[8];
            clist_pop_front(&fundcodes, next);
            printf("next %s, %d tasks left in queue.\n", next, fundcodes.n);
            clist_push_back(&inflight, next);
            pthread_cond_signal(&state_cv);
        } else {
            thread_count--;
            printf("no more task left @thread[%d]\n", id);
            pthread_mutex_unlock(&state_mu);
            break;
        }
        pthread_mutex_unlock(&state_mu);
    }
    printf("thread finished %d\n", thread_count);
    return NULL;
}

static void print_queue(const CodeList *l)
{
    if (l->n <= 0) {
        printf("no task in queue\n");
        return;
    }
    printf("%d: %s-%s\n", l->n, l->codes[0], l->codes[l->n - 1]);
    for (int i = 0; i < l->n; i++) {
        printf("%s%s", l->codes[i], (i + 1 == l->n) ? "\n" : ", ");
    }
}

static void spawn_worker(void)
{
    pthread_t th;
    int id = thread_count;
    if (pthread_create(&th, NULL, train_fund, (void *)(intptr_t)id) != 0) {
        perror("pthread_create");
        return;
    }
    pthread_detach(th);
    thread_count++;
}

static int parse_six(const char *s, char *out)
{
    if (!s || strlen(s) != 6) {
        return 0;
    }
    for (int i = 0; i < 6; i++) {
        if (s[i] < '0' || s[i] > '9') {
            return 0;
        }
    }
    memcpy(out, s, 7);
    return 1;
}

static void handle_cmd(char *cmd)
{
    while (*cmd && isspace((unsigned char)*cmd)) {
        cmd++;
    }
    size_t n = strlen(cmd);
    while (n > 0 && isspace((unsigned char)cmd[n - 1])) {
        cmd[--n] = 0;
    }
    if (n == 0) {
        return;
    }

    if (strcmp(cmd, "quit") == 0) {
        printf("quitting....\n");
        pthread_mutex_lock(&run_mu);
        for (int i = 0; i < ANN_MAX_RUNNING; i++) {
            if (g_running[i]) {
                g_running[i]->interrupt = 1;
            }
        }
        pthread_mutex_unlock(&run_mu);
        pthread_mutex_lock(&state_mu);
        Quit_flag = 1;
        pthread_cond_broadcast(&state_cv);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "list") == 0 || strcmp(cmd, "ls") == 0) {
        pthread_mutex_lock(&state_mu);
        print_queue(&fundcodes);
        if (processing.n > 0) {
            printf("processing:");
            for (int i = 0; i < processing.n; i++) {
                printf(" %s", processing.codes[i]);
            }
            printf("\n");
        }
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "reverse") == 0) {
        pthread_mutex_lock(&state_mu);
        clist_reverse(&fundcodes);
        print_queue(&fundcodes);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "shuffle") == 0) {
        uint32_t rng = (uint32_t)time(NULL);
        pthread_mutex_lock(&state_mu);
        clist_shuffle(&fundcodes, &rng);
        print_queue(&fundcodes);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "ascend") == 0) {
        pthread_mutex_lock(&state_mu);
        clist_sort(&fundcodes, 0);
        print_queue(&fundcodes);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "descend") == 0) {
        pthread_mutex_lock(&state_mu);
        clist_sort(&fundcodes, 1);
        print_queue(&fundcodes);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "+thread") == 0) {
        pthread_mutex_lock(&state_mu);
        if (fundcodes.n > 0) {
            char code[8];
            clist_pop_front(&fundcodes, code);
            clist_push_back(&inflight, code);
            printf("dispatch %s, %d tasks left in queue.\n", code, fundcodes.n);
        }
        spawn_worker();
        printf("Thread:%d\n", thread_count);
        pthread_cond_signal(&state_cv);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "-thread") == 0) {
        pthread_mutex_lock(&state_mu);
        ThreadReduce++;
        printf("Thread reduce:%d\n", ThreadReduce);
        pthread_cond_signal(&state_cv);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (strcmp(cmd, "status") == 0) {
        pthread_mutex_lock(&state_mu);
        printf("Task: %d/%d Thread count: %d, Queue: %d\n", thisTask, taskCount,
               thread_count, fundcodes.n);
        printf("Processing codes:");
        if (processing.n == 0) {
            printf(" set()\n");
        } else {
            printf(" {");
            for (int i = 0; i < processing.n; i++) {
                printf("'%s'%s", processing.codes[i],
                       i + 1 == processing.n ? "" : ", ");
            }
            printf("}\n");
        }
        pthread_mutex_unlock(&state_mu);
        return;
    }

    int parts = 0, task = 0;
    if (sscanf(cmd, "refuel %d %d", &parts, &task) == 2) {
        if (task == 0 && parts == 0) {
            task = thisTask;
            parts = taskCount;
        }
        if (task > 0 && parts > 0 && codepool.n >= parts) {
            CodeList slice;
            clist_init(&slice);
            slice_pool(&codepool, parts, task, &slice);
            pthread_mutex_lock(&state_mu);
            clist_extend_unique(&fundcodes, &slice, &fundcodes, &processing);
            printf("refuel: ");
            print_queue(&slice);
            pthread_mutex_unlock(&state_mu);
            thisTask = task;
            taskCount = parts;
        }
        return;
    }
    if (sscanf(cmd, "defuel %d %d", &parts, &task) == 2) {
        if (task > 0 && parts > 0 && codepool.n >= parts) {
            CodeList slice;
            clist_init(&slice);
            slice_pool(&codepool, parts, task, &slice);
            pthread_mutex_lock(&state_mu);
            printf("defuel: ");
            print_queue(&slice);
            clist_subtract(&fundcodes, &slice);
            pthread_mutex_unlock(&state_mu);
        }
        return;
    }
    char six[8];
    if (sscanf(cmd, "init %7s", six) == 1 && parse_six(six, six)) {
        train_one(six, train_epoch, train_lr);
        return;
    }
    int ep = 0;
    if (sscanf(cmd, "epoch %d", &ep) == 1 && ep > 0) {
        train_epoch = ep;
        printf("set epoch=%d\n", train_epoch);
        return;
    }
    if (sscanf(cmd, "p %7s", six) == 1 && parse_six(six, six)) {
        if (codepool.n <= 0) {
            printf("no code in code pool\n");
            return;
        }
        int ind = clist_index(&codepool, six);
        int g36 = pool_group_index(ind, codepool.n, 36);
        int g72 = pool_group_index(ind, codepool.n, 72);
        if (ind < 0) {
            printf("POS %s: N/A %d/36  %d/72", six, g36, g72);
        } else {
            printf("POS %s: %d %d/36  %d/72", six, ind + 1, g36, g72);
        }
        if (codepool.n >= taskCount && taskCount > 0) {
            CodeList slice;
            clist_init(&slice);
            slice_pool(&codepool, taskCount, thisTask, &slice);
            int ti = clist_index(&slice, six);
            if (ti >= 0) {
                printf("  task%d: %d/%d", thisTask, ti + 1, slice.n);
            } else {
                printf("  task%d: N/A", thisTask);
            }
        }
        printf("\n");
        return;
    }

    if (parse_six(cmd, six)) {
        pthread_mutex_lock(&state_mu);
        clist_push_front(&fundcodes, six);
        printf("add %s into queue.[%d]\n", six, fundcodes.n);
        pthread_cond_signal(&state_cv);
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (cmd[0] == '-' && parse_six(cmd + 1, six)) {
        pthread_mutex_lock(&state_mu);
        if (clist_remove(&fundcodes, six) == ANN_OK) {
            printf("remove %s from queue.[%d]\n", six, fundcodes.n);
        } else {
            printf("%s isn't in queue.[%d]\n", six, fundcodes.n);
        }
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (cmd[0] == '<' && parse_six(cmd + 1, six)) {
        pthread_mutex_lock(&state_mu);
        if (clist_remove(&fundcodes, six) == ANN_OK) {
            clist_push_front(&fundcodes, six);
            printf("move %s to front.[%d]\n", six, fundcodes.n);
        } else {
            printf("%s isn't in queue.[%d]\n", six, fundcodes.n);
        }
        pthread_mutex_unlock(&state_mu);
        return;
    }
    if (cmd[0] == '>' && parse_six(cmd + 1, six)) {
        pthread_mutex_lock(&state_mu);
        if (clist_remove(&fundcodes, six) == ANN_OK) {
            clist_push_back(&fundcodes, six);
            printf("move %s to rear.[%d]\n", six, fundcodes.n);
        } else {
            printf("%s isn't in queue.[%d]\n", six, fundcodes.n);
        }
        pthread_mutex_unlock(&state_mu);
        return;
    }
}

static int predict_one(const char *code)
{
    NavSeries nav;
    Dataset ds;
    Model m;
    memset(&nav, 0, sizeof(nav));
    memset(&ds, 0, sizeof(ds));
    memset(&m, 0, sizeof(m));
    if (db_open("./fund.db") != ANN_OK) {
        fprintf(stderr, "cannot open ./fund.db\n");
        return ANN_ERR;
    }
    if (model_create(&m, code, 0) != ANN_OK) {
        db_close();
        return ANN_ERR;
    }
    char binpath[256];
    weights_bin_path(code, binpath, sizeof(binpath));
    if (!path_exists(binpath) || weights_load(&m, binpath) != ANN_OK) {
        fprintf(stderr, "need %s (python3 c_annfund/tools/pth_to_bin.py ./ann/%s.pth)\n",
                binpath, code);
        model_destroy(&m);
        db_close();
        return ANN_ERR;
    }
    if (db_load_nav(code, ANN_NAV_PRED, &nav) != ANN_OK ||
        dataset_from_serial_pred(&nav, &ds) != ANN_OK) {
        fprintf(stderr, "pred dataset failed for %s\n", code);
        model_destroy(&m);
        nav_free(&nav);
        db_close();
        return ANN_ERR;
    }
    float *pred = (float *)malloc((size_t)ds.n * ANN_OUTSIZE * sizeof(float));
    if (!pred || model_predict(&m, &ds, pred) != ANN_OK) {
        free(pred);
        dataset_free(&ds);
        nav_free(&nav);
        model_destroy(&m);
        db_close();
        return ANN_ERR;
    }
    printf("code=%s n=%d\n", code, ds.n);
    int show = ds.n < 5 ? ds.n : 5;
    for (int i = ds.n - show; i < ds.n; i++) {
        printf("%d %.8f %.8f %.8f\n", i, pred[i * 3], pred[i * 3 + 1],
               pred[i * 3 + 2]);
    }
    free(pred);
    dataset_free(&ds);
    nav_free(&nav);
    model_destroy(&m);
    db_close();
    return ANN_OK;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[1], "--predict") == 0) {
        return predict_one(argv[2]) == ANN_OK ? 0 : 1;
    }

    ensure_ann_dir();
    clist_init(&codepool);
    clist_init(&fundcodes);
    clist_init(&processing);
    clist_init(&inflight);

    if (scan_ann_pool("./ann", &codepool) != ANN_OK) {
        fprintf(stderr, "scan ./ann failed (empty is ok if dir exists)\n");
    }
    char order_flag[8] = {0};
    if (argc >= 3) {
        taskCount = atoi(argv[1]);
        thisTask = atoi(argv[2]);
    }
    if (argc >= 4) {
        train_epoch = atoi(argv[3]);
    }
    if (argc >= 5) {
        snprintf(order_flag, sizeof(order_flag), "%s", argv[4]);
    }

    int totalCount = codepool.n;
    if (totalCount >= taskCount && taskCount > 0) {
        slice_pool(&codepool, taskCount, thisTask, &fundcodes);
        if (fundcodes.n > 0) {
            if (order_flag[0] == 'r') {
                clist_reverse(&fundcodes);
                printf("%s-%d-%s\n", fundcodes.codes[0], fundcodes.n,
                       fundcodes.codes[fundcodes.n - 1]);
            } else if (order_flag[0] == 's') {
                uint32_t rng = (uint32_t)time(NULL);
                clist_shuffle(&fundcodes, &rng);
                print_queue(&fundcodes);
            } else {
                printf("%s-%d-%s\n", fundcodes.codes[0], fundcodes.n,
                       fundcodes.codes[fundcodes.n - 1]);
            }
        }
    }

    if (db_open("./fund.db") != ANN_OK) {
        fprintf(stderr, "warning: cannot open ./fund.db\n");
    }

    thread_count = 0;
    int start_n = 1;
    if (taskCount <= 0) {
        start_n = thisTask;
    }
    for (int i = 0; i < start_n; i++) {
        pthread_mutex_lock(&state_mu);
        if (fundcodes.n > 0) {
            char code[8];
            clist_pop_front(&fundcodes, code);
            clist_push_back(&inflight, code);
        }
        spawn_worker();
        pthread_mutex_unlock(&state_mu);
    }

    char line[256];
    while (!Quit_flag) {
        printf(">>>:");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) {
            Quit_flag = 1;
            pthread_cond_broadcast(&state_cv);
            break;
        }
        handle_cmd(line);
    }
    /* wait a bit for workers to notice quit */
    sleep(1);
    db_close();
    printf("done\n");
    return 0;
}
