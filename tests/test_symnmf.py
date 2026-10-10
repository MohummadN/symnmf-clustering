"""Tests for the SymNMF C library, its Python bindings and the CLIs.

Run from the repository root after building:  make && make ext && pytest
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import symnmfmodule  # noqa: E402
from analysis import kmeans  # noqa: E402
from symnmf import cluster, init_h, symnmf  # noqa: E402

ERROR = "An Error Has Occurred"


# --------------------------------------------------------------------------
# NumPy reference implementations of the definitions in the README
# --------------------------------------------------------------------------

def ref_sym(X):
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    A = np.exp(-sq / 2)
    np.fill_diagonal(A, 0)
    return A


def ref_ddg(X):
    return np.diag(ref_sym(X).sum(axis=1))


def ref_norm(X):
    A = ref_sym(X)
    d = 1 / np.sqrt(A.sum(axis=1))
    return A * d[:, None] * d[None, :]


REFERENCES = {"sym": ref_sym, "ddg": ref_ddg, "norm": ref_norm}


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.5,
                      random_state=42)
    return X, y


@pytest.fixture
def points_file(tmp_path, blobs):
    path = tmp_path / "points.txt"
    np.savetxt(path, blobs[0], fmt="%.6f", delimiter=",")
    return path


def run(*args):
    return subprocess.run(list(map(str, args)), capture_output=True, text=True, cwd=ROOT)


def parse(output):
    return np.array([[float(v) for v in line.split(",")] for line in output.strip().splitlines()])


# --------------------------------------------------------------------------
# Matrix goals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("goal", REFERENCES)
def test_extension_matches_reference(goal, blobs):
    X = blobs[0]
    np.testing.assert_allclose(getattr(symnmfmodule, goal)(X), REFERENCES[goal](X), atol=1e-12)


@pytest.mark.parametrize("goal", REFERENCES)
def test_c_cli_matches_reference(goal, points_file):
    result = run("./symnmf", goal, points_file)
    assert result.returncode == 0, result.stderr
    X = np.loadtxt(points_file, delimiter=",")
    np.testing.assert_allclose(parse(result.stdout), REFERENCES[goal](X), atol=5e-5)


@pytest.mark.parametrize("goal", REFERENCES)
def test_c_and_python_cli_agree(goal, points_file):
    c_out = run("./symnmf", goal, points_file).stdout
    py_out = run(sys.executable, "symnmf.py", 3, goal, points_file).stdout
    assert c_out == py_out


def test_extension_accepts_lists_and_arrays(blobs):
    X = blobs[0][:10]
    assert symnmfmodule.sym(X.tolist()) == symnmfmodule.sym(X)


# --------------------------------------------------------------------------
# SymNMF optimisation
# --------------------------------------------------------------------------

def objective(W, H):
    return np.linalg.norm(W - H @ H.T) ** 2


def test_symnmf_decreases_objective_and_stays_non_negative(blobs):
    X = blobs[0]
    W = ref_norm(X)
    H0 = init_h(W, 3)
    H = symnmf(X, 3)
    assert H.shape == (len(X), 3)
    assert np.all(H >= 0)
    assert objective(W, H) < objective(W, H0)


def test_symnmf_respects_max_iter(blobs):
    W = ref_norm(blobs[0])
    H0 = init_h(W, 3)
    H1 = np.array(symnmfmodule.symnmf(W, H0, max_iter=1))
    beta = 0.5
    expected = H0 * (1 - beta + beta * (W @ H0) / (H0 @ H0.T @ H0))
    np.testing.assert_allclose(H1, expected, rtol=1e-12)


def test_symnmf_recovers_separated_clusters(blobs):
    X, y = blobs
    assert adjusted_rand_score(y, cluster(X, 3)) == 1.0


def test_symnmf_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        symnmfmodule.symnmf(np.eye(4), np.ones((3, 2)))


# --------------------------------------------------------------------------
# K-means baseline
# --------------------------------------------------------------------------

def test_kmeans_matches_sklearn(blobs):
    X = blobs[0]
    ours = kmeans(X, 3)
    theirs = KMeans(n_clusters=3, init=X[:3], n_init=1, tol=0).fit_predict(X)
    assert adjusted_rand_score(ours, theirs) == 1.0


# --------------------------------------------------------------------------
# Input parsing and error handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["1,2\n3,4\n5,6", "1,2\r\n3,4\r\n5,6\r\n", "1, 2\n3, 4\n5, 6\n\n"])
def test_c_cli_parses_file_variants(tmp_path, content):
    path = tmp_path / "in.txt"
    path.write_bytes(content.encode())
    result = run("./symnmf", "sym", path)
    assert result.returncode == 0
    np.testing.assert_allclose(parse(result.stdout), ref_sym(np.array([[1, 2], [3, 4], [5, 6]])), atol=5e-5)


@pytest.mark.parametrize("args", [
    ("sym", "does_not_exist.txt"),
    ("bogus", "examples/blobs_k4_d2.txt"),
    ("sym",),
])
def test_c_cli_errors(args):
    result = run("./symnmf", *args)
    assert result.returncode != 0
    assert ERROR in result.stdout + result.stderr


def test_c_cli_rejects_ragged_rows(tmp_path):
    path = tmp_path / "ragged.txt"
    path.write_text("1,2\n3\n")
    assert run("./symnmf", "sym", path).returncode != 0


def test_python_cli_error(tmp_path):
    result = run(sys.executable, "symnmf.py", 2, "symnmf", tmp_path / "missing.txt")
    assert result.returncode != 0
    assert result.stdout.strip() == ERROR
