# from distutils.core import setup
# from Cython.Build import cythonize
# from distutils.extension import Extension
# from Cython.Distutils import build_ext


# setup(
#     ext_modules = cythonize("monte_carlo_sec.pyx")
# )
# https://nealhughes.net/parallelcomp2/

# ext_modules=[
#     Extension("ray_trace_omp",
#               ["ray_trace_omp.pyx"],
#               libraries=["m"],
#               extra_compile_args=["-O3", "-ffast-math", "-march=native",
#                                       "-Xpreprocessor", "-fopenmp"],
#                 extra_link_args=["-Xpreprocessor", "-fopenmp"],
#               ) 
# ]

# setup( 
#   name = "ray_trace_omp",
#   cmdclass = {"build_ext": build_ext},
#   ext_modules = ext_modules
# )

# from setuptools import Extension, setup
# from Cython.Build import cythonize

# ext_modules = [
#     Extension(
#         "omp",
#         ["omp.pyx"],
#         extra_compile_args=["-O3", "-ffast-math", "-march=native","-Xpreprocessor", "-fopenmp"],
#         extra_link_args=["-Xpreprocessor", "-fopenmp"],
#     )
# ]

# setup(
#     name='mctwo',
#     ext_modules=cythonize("monte_carlo_2_omp.pyx"),
# )

# extra_compile_args=["-O3", "-ffast-math", "-march=native","-Xpreprocessor", "-fopenmp"],
# extra_link_args=["-Xpreprocessor", "-fopenmp"],
# ),

from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options



from Cython.Build import cythonize

# set gcc to gcc-12
import os
os.environ["CC"] = "gcc-12"

extensions = [
    Extension("mc_cython_seq", ["mc_cython_seq.pyx"]),
    Extension("mc_cython_omp", ["mc_cython_omp.pyx"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
    ),
    Extension("mc_cython_simd", ["mc_cython_simd.pyx"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
    ),
    Extension("omp_test", ["omp_test.pyx"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=cythonize(extensions),
)