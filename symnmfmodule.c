#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/*
 * Python bindings for the SymNMF C library.
 * Matrices are accepted as any sequence of sequences of numbers (lists,
 * tuples or NumPy arrays) and returned as lists of lists of floats.
 */

/* Converts a rectangular sequence of sequences into a C matrix. */
static double **to_c_matrix(PyObject *obj, int *rows, int *cols) {
    PyObject *seq, *row = NULL;
    double **matrix = NULL;
    Py_ssize_t i, j;

    if ((seq = PySequence_Fast(obj, "expected a 2D sequence")) == NULL)
        return NULL;
    *rows = (int)PySequence_Fast_GET_SIZE(seq);
    *cols = *rows > 0 ? (int)PySequence_Size(PySequence_Fast_GET_ITEM(seq, 0)) : 0;
    if (*rows <= 0 || *cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "matrix must be non-empty");
        goto fail;
    }
    if ((matrix = alloc_matrix(*rows, *cols)) == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    for (i = 0; i < *rows; i++) {
        row = PySequence_Fast(PySequence_Fast_GET_ITEM(seq, i), "expected a 2D sequence");
        if (row == NULL)
            goto fail;
        if (PySequence_Fast_GET_SIZE(row) != *cols) {
            PyErr_SetString(PyExc_ValueError, "all rows must have the same length");
            goto fail;
        }
        for (j = 0; j < *cols; j++) {
            matrix[i][j] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(row, j));
            if (PyErr_Occurred())
                goto fail;
        }
        Py_CLEAR(row);
    }
    Py_DECREF(seq);
    return matrix;

fail:
    Py_XDECREF(row);
    Py_DECREF(seq);
    free_matrix(matrix);
    return NULL;
}

/* Converts a C matrix into a list of lists. Does not free the C matrix. */
static PyObject *to_py_matrix(double **matrix, int rows, int cols) {
    PyObject *result = PyList_New(rows), *row, *value;
    int i, j;

    if (result == NULL)
        return NULL;
    for (i = 0; i < rows; i++) {
        if ((row = PyList_New(cols)) == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, row);
        for (j = 0; j < cols; j++) {
            if ((value = PyFloat_FromDouble(matrix[i][j])) == NULL) {
                Py_DECREF(result);
                return NULL;
            }
            PyList_SET_ITEM(row, j, value);
        }
    }
    return result;
}

/* Shared implementation of sym(), ddg() and norm(). */
static PyObject *points_to_matrix(PyObject *args, double **(*compute)(double **, int, int)) {
    PyObject *points, *result;
    double **X, **M;
    int N, d;

    if (!PyArg_ParseTuple(args, "O", &points))
        return NULL;
    if ((X = to_c_matrix(points, &N, &d)) == NULL)
        return NULL;
    M = compute(X, N, d);
    free_matrix(X);
    if (M == NULL)
        return PyErr_NoMemory();
    result = to_py_matrix(M, N, N);
    free_matrix(M);
    return result;
}

static PyObject *py_sym(PyObject *self, PyObject *args) {
    (void)self;
    return points_to_matrix(args, sym_matrix);
}

static PyObject *py_ddg(PyObject *self, PyObject *args) {
    (void)self;
    return points_to_matrix(args, ddg_matrix);
}

static PyObject *py_norm(PyObject *self, PyObject *args) {
    (void)self;
    return points_to_matrix(args, norm_matrix);
}

static PyObject *py_symnmf(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *keywords[] = {"W", "H0", "max_iter", "eps", NULL};
    PyObject *W_obj, *H0_obj, *result = NULL;
    double **W = NULL, **H0 = NULL, **H = NULL;
    int max_iter = SYMNMF_MAX_ITER, N, n_cols, n_rows, k;
    double eps = SYMNMF_EPSILON;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|id", keywords,
                                     &W_obj, &H0_obj, &max_iter, &eps))
        return NULL;
    if ((W = to_c_matrix(W_obj, &N, &n_cols)) == NULL)
        goto done;
    if ((H0 = to_c_matrix(H0_obj, &n_rows, &k)) == NULL)
        goto done;
    if (n_cols != N || n_rows != N) {
        PyErr_SetString(PyExc_ValueError, "W must be N x N and H0 must be N x k");
        goto done;
    }
    if ((H = symnmf_factorize(W, H0, N, k, max_iter, eps)) == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    result = to_py_matrix(H, N, k);

done:
    free_matrix(W);
    free_matrix(H0);
    free_matrix(H);
    return result;
}

static PyMethodDef methods[] = {
    {"symnmf", (PyCFunction)(void (*)(void))py_symnmf, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("symnmf(W, H0, max_iter=300, eps=1e-4)\n--\n\n"
               "Optimise H >= 0 minimising ||W - H H^T||_F^2, starting from H0.")},
    {"sym", py_sym, METH_VARARGS,
     PyDoc_STR("sym(X)\n--\n\nSimilarity matrix of the data points X.")},
    {"ddg", py_ddg, METH_VARARGS,
     PyDoc_STR("ddg(X)\n--\n\nDiagonal degree matrix of the data points X.")},
    {"norm", py_norm, METH_VARARGS,
     PyDoc_STR("norm(X)\n--\n\nNormalized similarity matrix of the data points X.")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    PyDoc_STR("C implementation of Symmetric NMF clustering."),
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&module);
}
