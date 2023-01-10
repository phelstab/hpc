# Import necessary libraries
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp
from cython.parallel cimport prange

# Set the number of iterations for the simulation
cdef int n_iter = 10000000

# Set the simulation parameters
cdef float mu = 0.1
cdef float sigma = 0.5
cdef float S0 = 100.0
cdef float T = 1.0
cdef float r = 0.05

# Set the number of independent paths to simulate
cdef int n_paths = 1000

# Allocate memory for the simulated paths
cdef np.ndarray[np.float32_t, ndim=2] paths = np.empty((n_paths, n_iter+1), dtype=np.float32)

# Set the initial price for all paths
paths[:, 0] = S0

# Set the step size for the simulation
cdef float dt = T / n_iter

# Set the constants for the simulation
cdef float sqrt_dt = sqrt(dt)
cdef float half_dt = 0.5 * dt
cdef float half_sigma_sq = 0.5 * sigma * sigma

# Use SIMD-intrinsics to speed up the simulation
cdef float32x4_t mu_simd, sigma_simd, half_dt_simd, sqrt_dt_simd, half_sigma_sq_simd
mu_simd = float32x4_t(mu, mu, mu, mu)
sigma_simd = float32x4_t(sigma, sigma, sigma, sigma)
half_dt_simd = float32x4_t(half_dt, half_dt, half_dt, half_dt)
sqrt_dt_simd = float32x4_t(sqrt_dt, sqrt_dt, sqrt_dt, sqrt_dt)
half_sigma_sq_simd = float32x4_t(half_sigma_sq, half_sigma_sq, half_sigma_sq, half_sigma_sq)

# Set the seed for the random number generator
np.random.seed(0)


with nogil, parallel(num_threads=4):
    cdef int i, j
    for i in prange(n_paths, nogil=True):
        cdef float32x4_t Z1, Z2, Z3, Z4
        cdef float32x4_t S1, S2, S3, S4
        S1 = float32x4_t(paths[i, 0], paths[i, 0], paths[i, 0], paths[i, 0])
        S2 = float32x4_t(paths[i, 0], paths[i, 0], paths[i, 0], paths[i, 0])
        S3 = float32x4_t(paths[i, 0], ppaths[i, 0], paths[i, 0], paths[i, 0])
        S4 = float32x4_t(paths[i, 0], paths[i, 0], paths[i, 0], paths[i, 0])