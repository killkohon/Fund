#include "db.h"
#include <pthread.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static sqlite3 *g_db = NULL;
static pthread_mutex_t db_mu = PTHREAD_MUTEX_INITIALIZER;

int db_open(const char *path)
{
    int rc;
    pthread_mutex_lock(&db_mu);
    if (g_db) {
        rc = ANN_OK;
        goto out;
    }
    if (sqlite3_open(path, &g_db) != SQLITE_OK) {
        fprintf(stderr, "sqlite open failed: %s\n", sqlite3_errmsg(g_db));
        g_db = NULL;
        rc = ANN_ERR;
        goto out;
    }
    rc = ANN_OK;
out:
    pthread_mutex_unlock(&db_mu);
    return rc;
}

void db_close(void)
{
    pthread_mutex_lock(&db_mu);
    if (g_db) {
        sqlite3_close(g_db);
        g_db = NULL;
    }
    pthread_mutex_unlock(&db_mu);
}

void nav_free(NavSeries *s)
{
    if (!s) {
        return;
    }
    if (s->vdate) {
        for (int i = 0; i < s->n; i++) {
            free(s->vdate[i]);
        }
        free(s->vdate);
    }
    free(s->unfixed);
    s->vdate = NULL;
    s->unfixed = NULL;
    s->n = 0;
}

int db_load_nav(const char *fundcode, int limit, NavSeries *out)
{
    int rc;
    pthread_mutex_lock(&db_mu);
    memset(out, 0, sizeof(*out));
    if (!g_db) {
        fprintf(stderr, "db_load_nav: database not open\n");
        rc = ANN_ERR;
        goto out;
    }
    sqlite3_stmt *st = NULL;
    const char *sql =
        "SELECT vdate, unfixedvalue FROM fund_values "
        "WHERE fundcode=? ORDER BY vdate DESC LIMIT ?";
    if (sqlite3_prepare_v2(g_db, sql, -1, &st, NULL) != SQLITE_OK) {
        rc = ANN_ERR;
        goto out;
    }
    sqlite3_bind_text(st, 1, fundcode, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(st, 2, limit);

    int cap = 0;
    char **dates = NULL;
    float *vals = NULL;
    int n = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
        const unsigned char *d = sqlite3_column_text(st, 0);
        if (!d || sqlite3_column_type(st, 1) == SQLITE_NULL) {
            continue;
        }
        if (n >= cap) {
            cap = cap ? cap * 2 : 64;
            dates = realloc(dates, (size_t)cap * sizeof(*dates));
            vals = realloc(vals, (size_t)cap * sizeof(*vals));
            if (!dates || !vals) {
                sqlite3_finalize(st);
                rc = ANN_ERR;
                goto out;
            }
        }
        dates[n] = strdup((const char *)d);
        vals[n] = (float)sqlite3_column_double(st, 1);
        n++;
    }
    sqlite3_finalize(st);

    /* Python inserts each DESC row at index 0 → chronological */
    for (int i = 0, j = n - 1; i < j; i++, j--) {
        char *td = dates[i];
        dates[i] = dates[j];
        dates[j] = td;
        float tv = vals[i];
        vals[i] = vals[j];
        vals[j] = tv;
    }
    out->vdate = dates;
    out->unfixed = vals;
    out->n = n;
    if (n <= 0) {
        fprintf(stderr, "db_load_nav: no rows for %s\n", fundcode);
        rc = ANN_ERR;
        goto out;
    }
    rc = ANN_OK;
out:
    pthread_mutex_unlock(&db_mu);
    return rc;
}
