#cython: language_level=3
# distutils: language = c++

from random_gauss cimport RandomGauss
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

# random_gauss.pyx
# cdef extern from "random_gauss.h":
#     cdef cppclass RandomGauss:
#         RandomGauss(double mean, double stddev)
#         double next()

# cdef extern from "random_gauss.h":
#     cdef cppclass RandomGauss:
#         long long get_current_time_ns()

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)


#get sqrt, cos, log from c extern
cdef extern from "math.h":
    double log(double x)
    double sqrt(double x)
    double cos(double x)


# cdef class PyRandomGauss:
#     cdef RandomGauss* cpp_obj

#     def __cinit__(self, double mean, double stddev):
#         self.cpp_obj = new RandomGauss(mean, stddev)

#     def next(self):
#         return self.cpp_obj.next()


# def draw_rndm_test():
#     cdef PyRandomGauss gauss = PyRandomGauss(0, 1)
#     return gauss.next()

# cdef double draw_rndm(double mu, double sigma):
#     cdef PyRandomGauss gauss = PyRandomGauss(mu, sigma)
#     cdef double rnd = gauss.next()
#     return rnd
    
# cdef get_current_time():
#     cdef long long cur_time = PyRandomGauss.get_current_time_ns()
#     return cur_time

# cpdef test_time():
#     cdef long long cur_time = get_current_time()
#     print("cur_time: ", cur_time)


# cdef long int* get_timestamps_c_array(int num_sim, int num_days):
#     cdef long int* timestamps = <long int*> malloc(num_sim*num_days+100 * sizeof(long int))
#     cdef int i
#     for i in range(num_sim*num_days+100):
#         timestamps[i] = int(time.perf_counter() * 1e9)
#     return timestamps

# cdef double rndm_normal_blazin_fast(double mu, double sigma, int seed):
#     cdef double rnd
#     cdef double pi = 3.14159265358979323846
#     srand48(seed)
#     rnd = mu + sigma * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48())
#     return rnd

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


cpdef run(close_list, int num_sim, int num_days, double last_price):
    cdef long int* timestamps = <long int*> malloc((num_sim*num_days+100) * sizeof(long int))
    cdef array.array n_close_list = array.array('d', close_list)
    returns = get_pct_change(n_close_list)
    cdef double vola = get_standard_deviation(returns)
    cdef double lp = last_price
    cdef int x, y
    cdef signed int iterator = num_sim*num_days
    cdef double* randoms = <double*>malloc((iterator * sizeof(double)) + 15)
    cdef double** sim_array = <double**>malloc((num_sim * sizeof(double*)) + 15)

    for x in range(num_sim):
        sim_array[x] = <double*>malloc((num_days * sizeof(double)) + 15)

    cdef double rnd
    cdef double pi = 3.14159265358979323846
    for x in range(num_sim*num_days):
        srand48(int(time.perf_counter() * 1e9))
        v = 0 + vola * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48())
        randoms[x] = v


    print("num_sim: ", num_sim)
    print("num_days: ", (num_days - 1))
    print("iterator: ", iterator)
    print("vola: ", vola)
    print("lp: ", lp)
    print("len(close_list): ", len(close_list))
    print("len(returns): ", len(returns))

    # create empty np array 1-dimensional, double precision making troubles
    #s_randoms = np.empty(num_sim*num_days*(np.dtype(np.double).itemsize * 2))
    s_randoms = np.empty(num_sim*num_days) #s, dtype=np.float32

    for x in range(num_sim*num_days):
        srand48(int(time.perf_counter() * 1e9))
        v = (0 + vola * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48()))
        s_randoms[x] = v

    # count time for this loop
    start = time.perf_counter()
    # double precision making troubles, so we use float32
    #s_a = parallel_outer_loop(s_randoms, np.double(last_price), np.int32(num_days), np.int32(num_sim))
    s_a = parallel_outer_loop(s_randoms, np.float32(last_price), np.int32(num_days), np.int32(num_sim))

    print("Time taken for opencl compute: ", (time.perf_counter() - start))

    return s_a
    


cpdef parallel_outer_loop(randoms, lp, num_days, num_sim):
    # Create a context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Allocate memory for the input and output arrays on the device
    randoms_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, randoms.nbytes)
    sim_array_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, (num_sim * (num_days-1) * sizeof(double)) + 15)
    #sim_array_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, (num_sim * (num_days-1) * (np.dtype(np.float32).itemsize * 2)) + 15)
    sim_array = np.empty((num_sim, num_days-1))

    # Copy the input data to the device
    cl.enqueue_copy(queue, randoms_buf, randoms)

    # Create the OpenCL kernel
    kernel = """
        __kernel void outer_loop(__global double* randoms, __global double* sim_array, float lp, int num_days, int num_sim) {
            int y = get_global_id(0);
            for(int x = 0; x < num_days - 1; x++) {
                if (x == 0) {
                    sim_array[y*(num_days-1) + x] = lp * (1 + randoms[y * num_days + x]);
                } else {
                    sim_array[y*(num_days-1) + x] = sim_array[y*(num_days-1) + (x-1)] * (1 + randoms[y * num_days + x]);
                }
            }
        }
    """

    # Compile the kernel
    prg = cl.Program(ctx, kernel).build()

    # Execute the kernel on the device
    global_size = (num_sim,)
    local_size = None
    prg.outer_loop(queue, global_size, local_size, randoms_buf, sim_array_buf, lp, num_days, num_sim)

    # Copy the result from the device to the host
    cl.enqueue_copy(queue, sim_array, sim_array_buf)
    return sim_array

    

# def generate_randoms(num_sim, num_days, vola, timestamps, randoms):
#     # Create OpenCL context and command queue
#     ctx = cl.create_some_context()
#     queue = cl.CommandQueue(ctx)

#     # Create input and output buffers
#     timestamps_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=timestamps)
#     randoms_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, randoms.nbytes)

#     # Create kernel
#     kernel = """
#          __kernel void random_numbers(__global ulong *timestamps, __global float *randoms, float vola) {
#             int gid = get_global_id(0);
#             double pi = 3.14159265358979323846;
#             unsigned int seed = (unsigned int)timestamps[gid];
#             float4 rand = intel_printf(&seed);
#             randoms[gid] = vola * sqrt(-2.0 * log(dot(rand, (float4)(1.0f,0.0f,0.0f,0.0f))) ) * cos(2 * pi * dot(rand, (float4)(1.0f,0.0f,0.0f,0.0f)));
#         }
#     """

#     # Compile kernel
#     prg = cl.Program(ctx, kernel).build()

#     # Execute kernel
#     global_size = (num_sim*num_days,)
#     local_size = None

#     prg.random_numbers(queue, global_size, local_size, timestamps_buf, randoms_buf)

#     # Read result
#     cl.enqueue_copy(queue, randoms, randoms_buf)
#     return randoms
