"""Configuration for the llm package."""

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


def get_cython_extension_modules() -> list[Extension]:
    """Compile Cython sources into extension modules"""
    return cythonize(
        [
            "llm/tokenizers/cython/stdtoken.pyx",
            "llm/tokenizers/cython/frequencies.pyx",
            "llm/tokenizers/cython/merge.pyx",
            "llm/tokenizers/cython/bpe.pyx",
        ],
        compiler_directives={
            "language_level": "3str",
        },
        build_dir="build",
    )


setup(
    name="llm",
    ext_modules=get_cython_extension_modules(),
    packages=find_packages(),
    zip_safe=False,
)
