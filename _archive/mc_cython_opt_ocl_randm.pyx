
#cython: language_level=3
# distutils: language = c++

import random
import math
from random import random as rand

import cython
from cpython cimport array
from cython cimport view
from libc.stdlib cimport malloc, free
import time
import pyopencl as cl
import numpy as np


"""Draw random samples from mean loc and standard deviation scale"""

from random_gauss cimport RandomGauss
cdef extern from "random_gauss.h":
    cdef cppclass RandomGauss:
        RandomGauss(double mean, double stddev)
        double next()

cdef class PyRandomGauss:
    cdef RandomGauss* cpp_obj
    def __cinit__(self, double mean, double stddev):
        self.cpp_obj = new RandomGauss(mean, stddev)

    def next(self):
        return self.cpp_obj.next()


#https://stackoverflow.com/a/32746954
#get generate random double from seed
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)


#get sqrt, cos, log from c extern
cdef extern from "math.h":
    double log(double x)
    double sqrt(double x)
    double cos(double x)

# pure in c (very fast) > pure python with math and random function (fast) > random.gauss (slow) > (very slow)  > numpy.random.normal (very very very slow)
cdef rndm_normal(mu, sigma):
    cdef double rnd = random.gauss(mu, sigma)
    return rnd
def rndm_normal_new(mu, sigma):
    rnd = mu + sigma * math.sqrt(-2.0 * math.log(random.random())) * math.cos(2 * math.pi * random.random())
    return rnd
cdef double rndm_normal_blazin_fast(double mu, double sigma):
    cdef double rnd
    cdef double pi = 3.14159265358979323846
    #getting the current time in nanoseconds with time.perf_counter() is not sexy enough (lazy to deal with c structs for now), but good enough for now
    cdef long int seed = int(time.perf_counter() * 1e9) 
    srand48(seed)
    rnd = mu + sigma * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48())
    return rnd


"""
    Draw random samples from mean loc and standard deviation scale. With Box Muller transform
    https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    Inspiration -> https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
"""

@cython.boundscheck(False)
cpdef double get_sum(double[:] series):
    cdef double _sum = 0.0
    cdef Py_ssize_t i
    cdef signed int n = 5790
    for i in range(n):
        _sum += series[i]
    return _sum

from libc.stdlib cimport malloc, free
@cython.boundscheck(False)
cdef double[:] get_differences_from_mean(double[:] series, double _mean):
    cdef signed int n = 5790
    cdef double* result = <double*>malloc(n * sizeof(double))
    cdef Py_ssize_t g
    for g in range(n):
        result[g] = (series[g] - _mean) ** 2
    cdef double[:] _deff = <double[:n]>result
    return _deff

#https://en.wikipedia.org/wiki/Standard_deviation
# we need series in python rest in c    
cpdef double get_standard_deviation(double[:] series):
    # Calculate the mean of the series
    cdef double _sum = 0.0
    cdef Py_ssize_t i
    cdef signed int n = 5790
    #print("ignore2: ", len(series))
    for i in range(n):
        _sum += series[i]
    cdef double _mean = _sum / n
    # Differences between each element and the mean
    cdef double x
    #differences = [(x - _mean) ** 2 for x in series]
    cdef double[:] differences = get_differences_from_mean(series, _mean)
    # Variance by dividing the sum of the differences by the length of the series
    cdef double _sumdiff = get_sum(differences)
    cdef double variance = _sumdiff / n
    # Standard deviation, square root of the variance
    cdef double std_dev = variance ** 0.5
    return std_dev



#Alt to: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
cpdef double[:] get_pct_change(double[:] series):
    cdef signed int n = 5791
    cdef Py_ssize_t i
    cdef double* pct_changes = <double*>malloc(n * sizeof(double))
    for i in range(1, n):
        pct_changes[i] = (series[i] - series[i-1]) / series[i-1]
    cdef double[:] _pct_changes = <double[:n]>pct_changes
    return _pct_changes

cdef convert_to_python(double **ptr, int n, int m):
    cdef int x
    cdef int i
    outerList=[[]]
    for x in range(n):
        innerLst=[]
        for i in range(m):
            innerLst.append(ptr[x][i])
        outerList.append(innerLst)
    free(ptr)
    return outerList


from array import array

cdef get_randoms(iterator):
    randoms_array = array("d", [0]*iterator)
    return memoryview(randoms_array).cast("B")


"""
    Main monte-carlo part
"""

# @cython.boundscheck(False)
cpdef run(close_list, int num_sim, int num_days, double last_price):
    # convert python list to c array
    cdef array.array n_close_list = array.array('d', close_list)
    # prepare data for sim
    returns = get_pct_change(n_close_list)
    cdef double vola = get_standard_deviation(returns)
    cdef double lp = last_price
    cdef int x, y
    cdef signed int iterator = num_sim*num_days
    cdef double* randoms = <double*>malloc((iterator * sizeof(double)) + 15)
    cdef double** sim_array = <double**>malloc((num_sim * sizeof(double*)) + 15)

    # malloc sim array size of days for each sim
    for x in range(num_sim):
        sim_array[x] = <double*>malloc((num_days * sizeof(double)) + 15)
    # create randoms for each sim and day and store in 1 dimension

    # try openCl here

    for x in range(num_sim*num_days):
        randoms[x] = rndm_normal_blazin_fast(0, vola)

    # convert the double* to a numpy array
    #data_array = get_randoms(randoms)


    #test = simulate(lp, data_array, num_days, num_sim)
    # for each simulation
    # num_sim: 1000
    # num_days: 5791
    # x = 0 to 5790
    # y = 0 to 1000
    # c = the random counter from 0 to 5790*1000
    # randoms[] containing 5790*1000 randoms for every price
    # test
    #DEF n = 1000000000       # total number of items to process (must be even)
    #DEF m = n // 2  # number of __m128d pairs needed for the items

    cdef PyRandomGauss gauss = PyRandomGauss(0, 1)
    print(gauss.next())

    print("num_sim: ", num_sim)
    print("num_days: ", (num_days - 1) // 2) # check, must be even
    print("iterator: ", iterator)
    print("vola: ", vola)
    print("lp: ", lp)
    print("len(close_list): ", len(close_list))
    print("len(returns): ", len(returns))

    for y in range(num_sim):
        # 1 simulation
        for x in range(num_days - 1):
            if x == 0:
                sim_array[y][0] = lp * (1 + randoms[y * num_days + x])
            else:
                sim_array[y][x] = sim_array[y][x-1] * (1 + randoms[y * num_days + x])

    #free(randoms)
    return convert_to_python(sim_array, num_sim, num_days) 

    import pyopencl as cl
import numpy as np

def simulate(lp, randoms, num_days, num_sim):
    # Create OpenCL context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Allocate memory on the device for the simulation array and random numbers
    sim_array = np.empty((num_sim, num_days))
    sim_array_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, sim_array.nbytes)
    randoms_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, randoms.nbytes)

    # Copy input data to the device
    cl.enqueue_copy(queue, sim_array_buf, sim_array)
    cl.enqueue_copy(queue, randoms_buf, randoms)

    # Create OpenCL kernel
    kernel = """
    __kernel void sim_array_kernel(__global float *sim_array, __global float *randoms, float lp, int num_days) {
        int gid = get_global_id(0);
        float sim_val = lp * (1 + randoms[gid]);
        sim_array[gid] = sim_val;
        for(int i = 1; i < num_days; i++){
            sim_val *= (1 + randoms[gid + i*num_sim]);
            sim_array[gid + i*num_sim] = sim_val;
        }
    }
    """

    # Compile the kernel
    prg = cl.Program(ctx, kernel).build()

    # Execute the kernel
    global_size = (num_sim,)
    local_size = None
    prg.sim_array_kernel(queue, global_size, local_size, sim_array_buf, randoms_buf, lp, num_days)

    # Copy the result from the device to the host
    cl.enqueue_copy(queue, sim_array, sim_array_buf)
    return sim_array
