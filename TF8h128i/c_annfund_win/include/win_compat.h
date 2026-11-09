#ifndef C_ANNFUND_WIN_COMPAT_H
#define C_ANNFUND_WIN_COMPAT_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
/* winsock2.h before windows.h - otherwise winsock.h collides */
#include <winsock2.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef SRWLOCK ann_mutex_t;
typedef CONDITION_VARIABLE ann_cond_t;

void ann_mutex_init(ann_mutex_t *m);
void ann_mutex_destroy(ann_mutex_t *m);
void ann_mutex_lock(ann_mutex_t *m);
void ann_mutex_unlock(ann_mutex_t *m);
/* 0 = acquired, nonzero = busy (same sense as pthread_mutex_trylock) */
int ann_mutex_trylock(ann_mutex_t *m);

void ann_cond_init(ann_cond_t *c);
void ann_cond_destroy(ann_cond_t *c);
void ann_cond_signal(ann_cond_t *c);
void ann_cond_broadcast(ann_cond_t *c);
int ann_cond_timedwait_ms(ann_cond_t *c, ann_mutex_t *m, DWORD ms);

int ann_thread_spawn(void *(*fn)(void *), void *arg);

void ann_sleep_ms(unsigned ms);
int ann_mkdir(const char *path);

int ann_net_init(void);
void ann_net_shutdown(void);
int ann_udp_send(const char *ipv4, unsigned port, const char *msg, int n);

#ifdef __cplusplus
}
#endif

#endif
