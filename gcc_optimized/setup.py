from setuptools import setup

from Cython.Build import cythonize
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options

extensions = [
    Extension("mc_gcc_opt", ["mc_gcc_opt.pyx"],
    ),
]




setup(
    ext_modules=cythonize(extensions),
)