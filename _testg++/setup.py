from setuptools import setup

from Cython.Build import cythonize
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options

extensions = [
    Extension("rect", ["rect.pyx"],
    ),
]




setup(
    ext_modules=cythonize(extensions),
)