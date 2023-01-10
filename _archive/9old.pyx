# Import necessary libraries
import random
from cython.parallel import prange, parallel

# Import SIMD intrinsics
from cython.parallel.simd.intrinsics import *

# Number of iterations for the Monte Carlo simulation
num_iterations = 1000000

# Initialize variables
total_in_circle = 0

# Seed the random number generator
random.seed()

# Use OpenMP to parallelize the loop
# Use SIMD intrinsics to vectorize the loop
# The "simd" directive tells Cython to vectorize the loop using SIMD intrinsics
@parallel(num_threads=4, simd=True)
def monte_carlo_simulation():
    global total_in_circle
    # Initialize a local variable to store the number of points in the circle for each thread
    local_in_circle = 0
    # Use the OpenMP "for" directive to parallelize the loop
    for i in prange(num_iterations, schedule='static', chunksize=1):
        # Generate random x and y coordinates in the range (-1, 1)
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        # Check if the point is in the circle
        if x * x + y * y <= 1:
            local_in_circle += 1
    # Use SIMD intrinsics to add the local variables together
    total_in_circle += horizontal_add(local_in_circle)

# Run the Monte Carlo simulation
monte_carlo_simulation()

# Calculate the value of pi
pi = 4 * total_in_circle / num_iterations
print(pi)
