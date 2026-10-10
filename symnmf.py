"""Symmetric Non-negative Matrix Factorization (SymNMF) clustering.

Usage:
    python3 symnmf.py <k> <symnmf|sym|ddg|norm> <input_file>

The heavy lifting is done by the C extension ``symnmfmodule``; this module
handles input, the random initialisation of H, and output formatting.
It can also be imported and used as a library (see :func:`cluster`).
"""

import argparse
import sys

import numpy as np

import symnmfmodule

SEED = 1234
MAX_ITER = 300
EPSILON = 1e-4
GOALS = ("symnmf", "sym", "ddg", "norm")


def read_points(path):
    """Load a comma-separated file of points into an (N, d) float array."""
    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=float)


def init_h(W, k, seed=SEED):
    """Randomly initialise H with values in [0, 2 * sqrt(mean(W) / k)]."""
    rng = np.random.RandomState(seed)
    upper = 2 * np.sqrt(np.mean(W) / k)
    return rng.uniform(0, upper, size=(len(W), k))


def symnmf(X, k, seed=SEED, max_iter=MAX_ITER, eps=EPSILON):
    """Return the (N, k) association matrix H for the data points X."""
    W = np.asarray(symnmfmodule.norm(X))
    H0 = init_h(W, k, seed)
    return np.asarray(symnmfmodule.symnmf(W, H0, max_iter=max_iter, eps=eps))


def cluster(X, k, **kwargs):
    """Hard-cluster X into k groups: each point joins its strongest column of H."""
    return np.argmax(symnmf(X, k, **kwargs), axis=1)


def compute_goal(goal, X, k):
    """Compute the matrix requested by the command-line ``goal``."""
    if goal == "symnmf":
        return symnmf(X, k)
    return np.asarray(getattr(symnmfmodule, goal)(X))


def format_matrix(matrix):
    """Format a matrix as comma-separated rows with 4 decimal places."""
    return "\n".join(",".join(f"{value:.4f}" for value in row) for row in matrix)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("k", type=int, help="number of clusters")
    parser.add_argument("goal", choices=GOALS, help="what to compute")
    parser.add_argument("file", help="comma-separated input points")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        X = read_points(args.file)
        if args.goal == "symnmf" and not 1 <= args.k < len(X):
            raise ValueError("k must satisfy 1 <= k < N")
        print(format_matrix(compute_goal(args.goal, X, args.k)))
    except Exception:
        print("An Error Has Occurred")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
