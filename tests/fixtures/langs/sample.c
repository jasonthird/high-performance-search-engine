#include <stdio.h>

struct widget { int width; };

static int compute_total(const int *items, int n) {
    int t = 0;
    for (int i = 0; i < n; i++) t += items[i];
    return t;
}
