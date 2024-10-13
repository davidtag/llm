from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            "llm/tokenizers/frequencies.pyx",
            "llm/tokenizers/merge.pyx",
            "llm/tokenizers/stdtoken.pyx",
        ],
        compiler_directives={
            # "boundscheck": False,
            # "wraparound": False,
            "language_level": "3str",
        },
    ),
    include_dirs=[numpy.get_include()],  # Include the numpy headers
)
