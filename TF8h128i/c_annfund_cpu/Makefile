CC ?= gcc
BASE_CFLAGS := -O2 -std=c11 -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=200809L
CFLAGS ?= $(BASE_CFLAGS)
LDFLAGS_COMMON := -lsqlite3 -lpthread -lm

# BLAS=openblas (default) | builtin | auto
# auto: use system OpenBLAS if cblas.h is found, else builtin naive SGEMM.
BLAS ?= openblas

OPENBLAS_INC := $(firstword \
	$(wildcard /usr/include/cblas.h) \
	$(wildcard /usr/include/openblas/cblas.h) \
	$(wildcard /usr/include/openblas-pthread/cblas.h) \
	$(wildcard /usr/local/include/openblas/cblas.h) \
	$(wildcard /usr/local/include/cblas.h))

ifeq ($(BLAS),auto)
  ifneq ($(OPENBLAS_INC),)
    BLAS := openblas
  else
    BLAS := builtin
  endif
endif

SRC = src/main.c src/db.c src/dataset.c src/queue.c src/weights.c src/weights_pth.c src/model.c
ifeq ($(BLAS),openblas)
  CFLAGS += -I$(dir $(OPENBLAS_INC))
  LDFLAGS = -lopenblas $(LDFLAGS_COMMON)
  $(info using OpenBLAS ($(OPENBLAS_INC)))
else ifeq ($(BLAS),builtin)
  CFLAGS += -DANN_BLAS_BUILTIN
  SRC += src/cblas_builtin.c
  LDFLAGS = $(LDFLAGS_COMMON)
  $(info using builtin SGEMM (no OpenBLAS); for speed later: enable EPEL and make BLAS=openblas)
else
  $(error BLAS must be auto, openblas, or builtin (got $(BLAS)))
endif

OBJS = $(SRC:.c=.o)

.PHONY: all clean test-bin-to-pth test-pth-to-bin

all: annfund

test-bin-to-pth: tools/test_bin_to_pth
test-pth-to-bin: tools/test_pth_to_bin

tools/test_bin_to_pth: tools/test_bin_to_pth.c src/weights_pth.c include/weights.h include/ann.h
	$(CC) $(CFLAGS) -o $@ tools/test_bin_to_pth.c src/weights_pth.c -lm

tools/test_pth_to_bin: tools/test_pth_to_bin.c src/weights_pth.c include/weights.h include/ann.h
	$(CC) $(CFLAGS) -o $@ tools/test_pth_to_bin.c src/weights_pth.c -lm

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

annfund: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f src/*.o annfund tools/test_gemm tools/test_bin_to_pth tools/test_pth_to_bin

tools/test_gemm: tools/test_gemm.c src/cblas_builtin.c
ifeq ($(BLAS),openblas)
	$(CC) $(CFLAGS) -o $@ tools/test_gemm.c $(LDFLAGS)
else
	$(CC) $(CFLAGS) -DANN_BLAS_BUILTIN -o $@ tools/test_gemm.c src/cblas_builtin.c $(LDFLAGS_COMMON)
endif
