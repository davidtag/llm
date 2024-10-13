"""Build the llm package."""

from setuptools import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        [
            "llm/tokenizers/cython/frequencies.pyx",
            "llm/tokenizers/cython/merge.pyx",
            "llm/tokenizers/cython/stdtoken.pyx",
        ],
        compiler_directives={
            "language_level": "3str",
        },
    ),
    include_dirs=[],
)
