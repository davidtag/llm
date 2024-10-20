"""Build the llm package.

From project root, run:
>> python setup.py build_ext --inplace

To run all tests
>> python setup.py build_ext --inplace && PYTHONPATH=. python -m unittest discover
"""

from setuptools import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        [
            "llm/tokenizers/cython/stdtoken.pyx",
            "llm/tokenizers/cython/frequencies.pyx",
            "llm/tokenizers/cython/merge.pyx",
            "llm/tokenizers/cython/bpe.pyx",
        ],
        compiler_directives={
            "language_level": "3str",
        },
    ),
    include_dirs=[],
)
