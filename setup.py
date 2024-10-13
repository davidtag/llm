"""Build the llm package.

From project root, run:
>> python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        [
            "llm/tokenizers/cython/stdtoken.pyx",
            "llm/tokenizers/cython/frequencies.pyx",
            "llm/tokenizers/cython/merge.pyx",
        ],
        compiler_directives={
            "language_level": "3str",
        },
    ),
    include_dirs=[],
)
