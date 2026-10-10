#ifndef SYMNMF_H
#define SYMNMF_H

/*
 * SymNMF - Symmetric Non-negative Matrix Factorization clustering.
 *
 * Matrices are represented as `double **` whose rows point into a single
 * contiguous block, so every matrix is released with one free_matrix() call.
 * Functions that allocate return NULL on failure and never terminate the
 * process, which makes the library safe to use from the Python extension.
 */

#define SYMNMF_BETA 0.5
#define SYMNMF_MAX_ITER 300
#define SYMNMF_EPSILON 1e-4

/**
 * Allocates a zero-initialised rows x cols matrix.
 * @return The matrix, or NULL on allocation failure or non-positive size.
 */
double **alloc_matrix(int rows, int cols);

/**
 * Releases a matrix allocated by alloc_matrix(). Accepts NULL.
 */
void free_matrix(double **matrix);

/**
 * Builds the similarity matrix A: a_ij = exp(-||x_i - x_j||^2 / 2), a_ii = 0.
 * @param X N x d data points.
 * @return N x N matrix A, or NULL on allocation failure.
 */
double **sym_matrix(double **X, int N, int d);

/**
 * Builds the diagonal degree matrix D, where d_ii is the i-th row sum of A.
 * @param X N x d data points.
 * @return N x N matrix D, or NULL on allocation failure.
 */
double **ddg_matrix(double **X, int N, int d);

/**
 * Builds the normalized similarity matrix W = D^-1/2 * A * D^-1/2.
 * @param X N x d data points.
 * @return N x N matrix W, or NULL on allocation failure.
 */
double **norm_matrix(double **X, int N, int d);

/**
 * Finds H >= 0 minimising ||W - H H^T||_F^2 with the multiplicative update
 *   H <- H * (1 - beta + beta * (W H) / (H H^T H)).
 * Stops after max_iter iterations or when ||H_next - H||_F^2 < eps.
 * @param W   N x N normalized similarity matrix.
 * @param H0  N x k non-negative initial guess (not modified).
 * @return N x k matrix H, or NULL on allocation failure.
 */
double **symnmf_factorize(double **W, double **H0, int N, int k,
                          int max_iter, double eps);

/**
 * Reads a comma-separated file of points (one point per line).
 * @param path Path to the input file.
 * @param N    Out: number of points.
 * @param d    Out: dimension of each point.
 * @return N x d matrix, or NULL if the file is missing, empty or malformed.
 */
double **read_points(const char *path, int *N, int *d);

/**
 * Prints a matrix as comma-separated rows with 4 decimal places.
 */
void print_matrix(double **matrix, int rows, int cols);

#endif /* SYMNMF_H */
