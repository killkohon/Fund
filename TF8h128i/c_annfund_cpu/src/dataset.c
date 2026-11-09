#include "dataset.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int parse_ymd(const char *s, int *y, int *m, int *d)
{
    return sscanf(s, "%d-%d-%d", y, m, d) == 3;
}

/* Civil from days algorithm (Howard Hinnant), days since 1970-01-01 */
static int days_from_civil(int y, unsigned m, unsigned d)
{
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int)doe - 719468;
}

static int weekday_mon0(int y, int m, int d)
{
    /* 1970-01-01 was Thursday. Python weekday: Mon=0 */
    int days = days_from_civil(y, (unsigned)m, (unsigned)d);
    int w = (days + 3) % 7; /* Thu=3 → Mon=0 */
    if (w < 0) {
        w += 7;
    }
    return w;
}

static void date_features(const NavSeries *nav, float *deltas, float *weekdays)
{
    const int y0 = 2020, m0 = 1, d0 = 1;
    int base = days_from_civil(y0, m0, d0);
    for (int i = 0; i < nav->n; i++) {
        int y, m, d;
        if (!parse_ymd(nav->vdate[i], &y, &m, &d)) {
            deltas[i] = 0;
            weekdays[i] = 0;
            continue;
        }
        deltas[i] = (float)(days_from_civil(y, (unsigned)m, (unsigned)d) - base) / 365.25f;
        weekdays[i] = (float)weekday_mon0(y, m, d) / 6.0f;
    }
}

void dataset_free(Dataset *ds)
{
    if (!ds) {
        return;
    }
    free(ds->x);
    free(ds->y);
    ds->x = NULL;
    ds->y = NULL;
    ds->n = 0;
}

int dataset_from_serial_train(const NavSeries *nav, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    const int isize = ANN_SERIAL_ISIZE;
    const int osize = ANN_SERIAL_OSIZE;
    if (nav->n < isize + osize) {
        fprintf(stderr, "series length %d < required %d\n", nav->n, isize + osize);
        return ANN_ERR;
    }
    float *deltas = malloc((size_t)nav->n * sizeof(float));
    float *wdays = malloc((size_t)nav->n * sizeof(float));
    if (!deltas || !wdays) {
        free(deltas);
        free(wdays);
        return ANN_ERR;
    }
    date_features(nav, deltas, wdays);

    int ns = nav->n - isize - osize + 1;
    ds->x = malloc((size_t)ns * ANN_INSIZE * sizeof(float));
    ds->y = malloc((size_t)ns * ANN_OUTSIZE * sizeof(float));
    if (!ds->x || !ds->y) {
        free(deltas);
        free(wdays);
        dataset_free(ds);
        return ANN_ERR;
    }
    for (int i = 0; i < ns; i++) {
        float *row = ds->x + (size_t)i * ANN_INSIZE;
        memcpy(row, nav->unfixed + i, (size_t)isize * sizeof(float));
        row[isize] = deltas[i + isize - 1];
        row[isize + 1] = wdays[i + isize - 1];
        const float *out = nav->unfixed + i + isize;
        float mx = out[0], mn = out[0], sm = 0;
        for (int k = 0; k < osize; k++) {
            float v = out[k];
            if (v > mx) {
                mx = v;
            }
            if (v < mn) {
                mn = v;
            }
            sm += v;
        }
        ds->y[i * 3 + 0] = mx;
        ds->y[i * 3 + 1] = sm / (float)osize;
        ds->y[i * 3 + 2] = mn;
    }
    ds->n = ns;
    ds->has_y = 1;
    free(deltas);
    free(wdays);
    printf("SerialDataset: %d samples\n", ns);
    printf("Input shape: (%d, %d), Output shape: (%d, 3)\n", ns, ANN_INSIZE, ns);
    return ANN_OK;
}

int dataset_from_serial_pred(const NavSeries *nav, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    const int insize = ANN_INSIZE;
    if (nav->n < insize) {
        return ANN_ERR;
    }
    float *deltas = malloc((size_t)nav->n * sizeof(float));
    float *wdays = malloc((size_t)nav->n * sizeof(float));
    if (!deltas || !wdays) {
        free(deltas);
        free(wdays);
        return ANN_ERR;
    }
    date_features(nav, deltas, wdays);
    int ns = nav->n - insize + 2;
    ds->x = malloc((size_t)ns * ANN_INSIZE * sizeof(float));
    if (!ds->x) {
        free(deltas);
        free(wdays);
        return ANN_ERR;
    }
    for (int i = 0; i < ns; i++) {
        float *row = ds->x + (size_t)i * ANN_INSIZE;
        memcpy(row, nav->unfixed + i, (size_t)(insize - 2) * sizeof(float));
        row[insize - 2] = deltas[i + insize - 2];
        row[insize - 1] = wdays[i + insize - 2];
    }
    ds->n = ns;
    ds->has_y = 0;
    free(deltas);
    free(wdays);
    return ANN_OK;
}

void dataset_shuffle_idx(int *idx, int n, uint32_t *rng)
{
    uint32_t s = *rng ? *rng : 1u;
    for (int i = 0; i < n; i++) {
        idx[i] = i;
    }
    for (int i = n - 1; i > 0; i--) {
        s = s * 1664525u + 1013904223u;
        int j = (int)(s % (uint32_t)(i + 1));
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
    *rng = s;
}
