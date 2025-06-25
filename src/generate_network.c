#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// Write one compare-swap line
void generate_compare_swap(FILE *out, size_t i, size_t j, bool direction) {
    fprintf(out, "\tif ((a[%lu] > a[%lu]) == %d) {\n", i, j, (int) direction);
    fprintf(out, "\t\tint tmp = a[%lu]; a[%lu] = a[%lu]; a[%lu] = tmp;\n", i, i, j, j);
    fprintf(out, "\t}\n");
}

void bitonic_merge(FILE *out, size_t lo, size_t n, bool direction) {
    if (n > 1) {
        size_t m = n / 2;
        for (int i = lo; i < lo + m; i++) {
            generate_compare_swap(out, i, i + m, direction);
        }
        bitonic_merge(out, lo, m, direction);
        bitonic_merge(out, lo + m, m, direction);
    }
}

void bitonic_sort(FILE *out, size_t lo, size_t n, bool direction) {
    if (n > 1) {
        size_t m = n / 2;
        bitonic_sort(out, lo, m, 1);
        bitonic_sort(out, lo + m, m, 0);
        bitonic_merge(out, lo, n, direction);
    }
}

// Generates the entire function to the file referenced by 'fd'
void generate_bitonic_sort_network(int fd, int N) {
    FILE *out = fdopen(fd, "w");
    if (!out) {
        perror("fdopen failed");
        exit(EXIT_FAILURE);
    }
    fprintf(out, "#include <stdint.h>\n");
    fprintf(out, "void bitonic_sort(uint32_t *a) {\n");
    bitonic_sort(out, 0, N, true);
    fprintf(out, "}\n");

    fflush(out);
    fclose(out);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0 || (N & (N - 1)) != 0) {
        fprintf(stderr, "Error: N must be a power of two.\n");
        return 1;
    }

    FILE *f = fopen("./sort_network/sort_newtork.c", "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    generate_bitonic_sort_network(fileno(f), N);

    fclose(f);
    return 0;
}