#include "win_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <ws2tcpip.h>

void ann_mutex_init(ann_mutex_t *m) { InitializeSRWLock(m); }
void ann_mutex_destroy(ann_mutex_t *m) { (void)m; }
void ann_mutex_lock(ann_mutex_t *m) { AcquireSRWLockExclusive(m); }
void ann_mutex_unlock(ann_mutex_t *m) { ReleaseSRWLockExclusive(m); }
int ann_mutex_trylock(ann_mutex_t *m)
{
    return TryAcquireSRWLockExclusive(m) ? 0 : 1;
}

void ann_cond_init(ann_cond_t *c) { InitializeConditionVariable(c); }
void ann_cond_destroy(ann_cond_t *c) { (void)c; }
void ann_cond_signal(ann_cond_t *c) { WakeConditionVariable(c); }
void ann_cond_broadcast(ann_cond_t *c) { WakeAllConditionVariable(c); }
int ann_cond_timedwait_ms(ann_cond_t *c, ann_mutex_t *m, DWORD ms)
{
    return SleepConditionVariableSRW(c, m, ms, 0) ? 0 : 1;
}

typedef struct {
    void *(*fn)(void *);
    void *arg;
} AnnThreadArg;

static DWORD WINAPI ann_thread_thunk(LPVOID p)
{
    AnnThreadArg *a = (AnnThreadArg *)p;
    void *(*fn)(void *) = a->fn;
    void *arg = a->arg;
    free(a);
    fn(arg);
    return 0;
}

int ann_thread_spawn(void *(*fn)(void *), void *arg)
{
    AnnThreadArg *a = (AnnThreadArg *)malloc(sizeof(*a));
    HANDLE th;
    if (!a) {
        return -1;
    }
    a->fn = fn;
    a->arg = arg;
    th = CreateThread(NULL, 0, ann_thread_thunk, a, 0, NULL);
    if (!th) {
        free(a);
        return -1;
    }
    CloseHandle(th);
    return 0;
}

void ann_sleep_ms(unsigned ms) { Sleep(ms); }

int ann_mkdir(const char *path)
{
    if (CreateDirectoryA(path, NULL)) {
        return 0;
    }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        return 0;
    }
    return -1;
}

static int g_wsa_ok = 0;
static SOCKET g_udp = INVALID_SOCKET;

int ann_net_init(void)
{
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return -1;
    }
    g_wsa_ok = 1;
    return 0;
}

void ann_net_shutdown(void)
{
    if (g_udp != INVALID_SOCKET) {
        closesocket(g_udp);
        g_udp = INVALID_SOCKET;
    }
    if (g_wsa_ok) {
        WSACleanup();
        g_wsa_ok = 0;
    }
}

int ann_udp_send(const char *ipv4, unsigned port, const char *msg, int n)
{
    struct sockaddr_in addr;
    u_long nb = 1;
    if (!g_wsa_ok || n <= 0) {
        return -1;
    }
    if (g_udp == INVALID_SOCKET) {
        g_udp = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (g_udp == INVALID_SOCKET) {
            return -1;
        }
        ioctlsocket(g_udp, FIONBIO, &nb);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if (inet_pton(AF_INET, ipv4, &addr.sin_addr) != 1) {
        return -1;
    }
    (void)sendto(g_udp, msg, n, 0, (struct sockaddr *)&addr, sizeof(addr));
    return 0;
}
