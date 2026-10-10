from setuptools import Extension, setup

setup(
    name="symnmf",
    version="1.1.0",
    description="Symmetric Non-negative Matrix Factorization clustering (C extension)",
    py_modules=["symnmf"],
    ext_modules=[
        Extension(
            "symnmfmodule",
            sources=["symnmfmodule.c", "symnmf.c"],
            extra_compile_args=["-O2"],
        )
    ],
)
