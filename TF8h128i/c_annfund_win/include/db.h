#ifndef C_ANNFUND_DB_H
#define C_ANNFUND_DB_H

#include "ann.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char **vdate;
    float *unfixed;
    int n;
} NavSeries;

int db_open(const char *path);
void db_close(void);
int db_load_nav(const char *fundcode, int limit, NavSeries *out);
void nav_free(NavSeries *s);

#ifdef __cplusplus
}
#endif

#endif
