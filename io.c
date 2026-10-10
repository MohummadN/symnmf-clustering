#include <stdio.h>
#include <stdlib.h>
#include "symnmf.h"

/* Growable array of doubles used while parsing the input file. */
typedef struct {
    double *data;
    size_t size;
    size_t capacity;
} Buffer;

static int buffer_push(Buffer *buf, double value) {
    double *grown;

    if (buf->size == buf->capacity) {
        buf->capacity = buf->capacity ? buf->capacity * 2 : 1024;
        grown = realloc(buf->data, buf->capacity * sizeof(double));
        if (grown == NULL)
            return 0;
        buf->data = grown;
    }
    buf->data[buf->size++] = value;
    return 1;
}

/* Returns the next character after a number, skipping spaces and '\r'. */
static int next_separator(FILE *file) {
    int ch;

    do {
        ch = fgetc(file);
    } while (ch == ' ' || ch == '\t' || ch == '\r');
    return ch;
}

/*
 * Reads every number in the file into buf. Validates that all rows have the
 * same number of columns, which is stored in *d. Returns the row count, or
 * -1 on a malformed file.
 */
static int parse_file(FILE *file, Buffer *buf, int *d) {
    double value;
    int rows = 0, cols = 0, ch;

    *d = 0;
    while (fscanf(file, "%lf", &value) == 1) {
        if (!buffer_push(buf, value))
            return -1;
        cols++;
        ch = next_separator(file);
        if (ch == ',')
            continue;
        if (ch != '\n' && ch != EOF)
            return -1;
        if (*d == 0)
            *d = cols;
        if (cols != *d)
            return -1;
        rows++;
        cols = 0;
    }
    return (cols == 0 && feof(file)) ? rows : -1;
}

double **read_points(const char *path, int *N, int *d) {
    Buffer buf = {NULL, 0, 0};
    FILE *file = fopen(path, "r");
    double **X = NULL;
    size_t i;

    if (file == NULL)
        return NULL;
    *N = parse_file(file, &buf, d);
    fclose(file);
    if (*N > 0 && (X = alloc_matrix(*N, *d)) != NULL)
        for (i = 0; i < buf.size; i++)
            X[0][i] = buf.data[i]; /* rows are contiguous */
    free(buf.data);
    return X;
}

void print_matrix(double **matrix, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            printf(j + 1 < cols ? "%.4f," : "%.4f\n", matrix[i][j]);
    }
}
