#include "weights.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_bin_to_pth <file.bin>\n");
        return 2;
    }
    if (weights_export_pth(argv[1]) != ANN_OK) {
        return 1;
    }
    return 0;
}
