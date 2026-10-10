#include <math.h>
#include <stdlib.h>
#include "symnmf.h"

/* ------------------------------------------------------------------------ */
/* Memory helpers                                                           */
/* ------------------------------------------------------------------------ */

double **alloc_matrix(int rows, int cols) {
    double **matrix;
    double *block;
    int i;

    if (rows <= 0 || cols <= 0)
        return NULL;
    matrix = malloc((size_t)rows * sizeof(double *));
    block = calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (matrix == NULL || block == NULL) {
        free(matrix);
        free(block);
        return NULL;
    }
    for (i = 0; i < rows; i++)
        matrix[i] = block + (size_t)i * (size_t)cols;
    return matrix;
}

void free_matrix(double **matrix) {
    if (matrix == NULL)
        return;
    free(matrix[0]);
    free(matrix);
}

/* ------------------------------------------------------------------------ */
/* Similarity, degree and normalized similarity                             */
/* ------------------------------------------------------------------------ */

/* Squared Euclidean distance between two d-dimensional points. */
static double squared_distance(const double *p, const double *q, int d) {
    double sum = 0.0, diff;
    int i;

    for (i = 0; i < d; i++) {
        diff = p[i] - q[i];
        sum += diff * diff;
    }
    return sum;
}

double **sym_matrix(double **X, int N, int d) {
    double **A = alloc_matrix(N, N);
    int i, j;

    if (A == NULL)
        return NULL;
    /* A is symmetric with a zero diagonal (already zeroed by calloc). */
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            A[i][j] = exp(-squared_distance(X[i], X[j], d) / 2.0);
            A[j][i] = A[i][j];
        }
    }
    return A;
}

/* Fills degrees[i] with the i-th row sum of the N x N matrix A. */
static void row_sums(double **A, int N, double *degrees) {
    int i, j;

    for (i = 0; i < N; i++) {
        degrees[i] = 0.0;
        for (j = 0; j < N; j++)
            degrees[i] += A[i][j];
    }
}

double **ddg_matrix(double **X, int N, int d) {
    double **A = sym_matrix(X, N, d);
    double **D = alloc_matrix(N, N);
    double *degrees = malloc((size_t)N * sizeof(double));
    int i;

    if (A == NULL || D == NULL || degrees == NULL) {
        free_matrix(A);
        free_matrix(D);
        free(degrees);
        return NULL;
    }
    row_sums(A, N, degrees);
    for (i = 0; i < N; i++)
        D[i][i] = degrees[i];
    free_matrix(A);
    free(degrees);
    return D;
}

double **norm_matrix(double **X, int N, int d) {
    double **W = sym_matrix(X, N, d);
    double *inv_sqrt = malloc((size_t)N * sizeof(double));
    int i, j;

    if (W == NULL || inv_sqrt == NULL) {
        free_matrix(W);
        free(inv_sqrt);
        return NULL;
    }
    /* D is diagonal, so D^-1/2 A D^-1/2 is just a_ij / sqrt(d_i * d_j). */
    row_sums(W, N, inv_sqrt);
    for (i = 0; i < N; i++)
        inv_sqrt[i] = inv_sqrt[i] > 0.0 ? 1.0 / sqrt(inv_sqrt[i]) : 0.0;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            W[i][j] *= inv_sqrt[i] * inv_sqrt[j];
    free(inv_sqrt);
    return W;
}

/* ------------------------------------------------------------------------ */
/* SymNMF optimisation                                                      */
/* ------------------------------------------------------------------------ */

/* out (r x c) = A (r x m) * B (m x c). */
static void multiply(double **A, double **B, double **out, int r, int m, int c) {
    int i, j, l;

    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++)
            out[i][j] = 0.0;
        for (l = 0; l < m; l++)
            for (j = 0; j < c; j++)
                out[i][j] += A[i][l] * B[l][j];
    }
}

/* out (k x k) = H^T H for an N x k matrix H. */
static void gram(double **H, double **out, int N, int k) {
    int i, a, b;

    for (a = 0; a < k; a++)
        for (b = 0; b < k; b++)
            out[a][b] = 0.0;
    for (i = 0; i < N; i++)
        for (a = 0; a < k; a++)
            for (b = 0; b < k; b++)
                out[a][b] += H[i][a] * H[i][b];
}

/*
 * One multiplicative update step: writes the next H into `next` and returns
 * ||next - H||_F^2. H H^T H is computed as H (H^T H), which costs O(N k^2)
 * instead of O(N^2 k) and never materialises an N x N matrix.
 */
static double update_step(double **W, double **H, double **next, double **WH,
                          double **HtH, double **HHtH, int N, int k) {
    double diff = 0.0, delta;
    int i, j;

    multiply(W, H, WH, N, N, k);
    gram(H, HtH, N, k);
    multiply(H, HtH, HHtH, N, k, k);
    for (i = 0; i < N; i++) {
        for (j = 0; j < k; j++) {
            next[i][j] = H[i][j] * (1.0 - SYMNMF_BETA
                                    + SYMNMF_BETA * WH[i][j] / HHtH[i][j]);
            delta = next[i][j] - H[i][j];
            diff += delta * delta;
        }
    }
    return diff;
}

double **symnmf_factorize(double **W, double **H0, int N, int k,
                          int max_iter, double eps) {
    double **H = alloc_matrix(N, k), **next = alloc_matrix(N, k);
    double **WH = alloc_matrix(N, k), **HHtH = alloc_matrix(N, k);
    double **HtH = alloc_matrix(k, k), **tmp;
    int i, j, iter;

    if (H == NULL || next == NULL || WH == NULL || HHtH == NULL || HtH == NULL) {
        free_matrix(H);
        H = NULL;
    } else {
        for (i = 0; i < N; i++)
            for (j = 0; j < k; j++)
                H[i][j] = H0[i][j];
        for (iter = 0; iter < max_iter; iter++) {
            double diff = update_step(W, H, next, WH, HtH, HHtH, N, k);
            tmp = H;
            H = next;
            next = tmp;
            if (diff < eps)
                break;
        }
    }
    free_matrix(next);
    free_matrix(WH);
    free_matrix(HHtH);
    free_matrix(HtH);
    return H;
}
