#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symnmf.h"

/*
 * Command-line interface:  ./symnmf <sym|ddg|norm> <input_file>
 * Prints the requested N x N matrix as comma-separated rows.
 */

typedef double **(*MatrixGoal)(double **X, int N, int d);

static MatrixGoal goal_function(const char *goal) {
    if (strcmp(goal, "sym") == 0)
        return sym_matrix;
    if (strcmp(goal, "ddg") == 0)
        return ddg_matrix;
    if (strcmp(goal, "norm") == 0)
        return norm_matrix;
    return NULL;
}

static int fail(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    return EXIT_FAILURE;
}

int main(int argc, char **argv) {
    MatrixGoal compute;
    double **X, **result;
    int N, d;

    if (argc != 3 || (compute = goal_function(argv[1])) == NULL)
        return fail();
    if ((X = read_points(argv[2], &N, &d)) == NULL)
        return fail();
    result = compute(X, N, d);
    free_matrix(X);
    if (result == NULL)
        return fail();
    print_matrix(result, N, N);
    free_matrix(result);
    return EXIT_SUCCESS;
}
