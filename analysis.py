"""Compare SymNMF clustering against K-means using the silhouette score.

Usage:
    python3 analysis.py <k> <input_file> [--plot output.png]
"""

import argparse
import sys

import numpy as np
from sklearn.metrics import silhouette_score

from symnmf import EPSILON, MAX_ITER, cluster, read_points


def kmeans(X, k, max_iter=MAX_ITER, eps=EPSILON):
    """Lloyd's K-means, initialised with the first k points.

    Stops when every centroid moves less than ``eps`` or after ``max_iter``
    iterations. Empty clusters get a zero centroid.
    """
    centroids = X[:k].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_centroids[j] = members.mean(axis=0)
        shift = np.linalg.norm(new_centroids - centroids, axis=1)
        centroids = new_centroids
        if np.all(shift < eps):
            break
    return labels


def score(X, labels):
    """Silhouette score, or NaN when fewer than two clusters were found."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return silhouette_score(X, labels)


def plot(X, results, path):
    """Save a side-by-side scatter plot of each method's clustering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if X.shape[1] > 2:  # project onto the two principal components
        centered = X - X.mean(axis=0)
        X = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    for ax, (name, (labels, value)) in zip(axes, results.items()):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=12)
        ax.set_title(f"{name}  (silhouette = {value:.4f})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("k", type=int, help="number of clusters")
    parser.add_argument("file", help="comma-separated input points")
    parser.add_argument("--plot", metavar="PNG", help="save a comparison plot")
    args = parser.parse_args(argv)

    try:
        X = read_points(args.file)
        results = {}
        for name, labels in (("nmf", cluster(X, args.k)), ("kmeans", kmeans(X, args.k))):
            results[name] = (labels, score(X, labels))
            print(f"{name}: {results[name][1]:.4f}")
        if args.plot:
            plot(X, {"SymNMF" if n == "nmf" else "K-means": r for n, r in results.items()},
                 args.plot)
    except Exception:
        print("An Error Has Occurred")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
