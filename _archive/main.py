from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# ext_modules = [
#     Extension("mymodule1",  ["mymodule1.py"]),
#     Extension("mymodule2",  ["mymodule2.py"]),
# ]

setup(
    name = 'ray_trace',
    cmdclass = {'build_ext': build_ext},
    #ext_modules = ext_modules
)






# hello = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# hello2 = ["hello", "world"]

# run.show_all(hello2)


# Monte Carlo simulation in Cython using SIMD-intrinsics and multi-threading

# from distutils.core import setup
# from Cython.Build import cythonize

# setup(
#     ext_modules = cythonize("monte_carlo.pyx")
# )

# print(monte_carlo.monte_carlo_pi(1000000))