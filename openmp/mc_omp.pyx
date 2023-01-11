#cython: language_level=3

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
from cython cimport floating
from cython cimport boundscheck, wraparound
cimport openmp
# import prange
from cython.parallel import prange

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

#get sqrt, cos, log from c extern
cdef extern from "math.h":
    double log(double x)
    double sqrt(double x)
    double cos(double x)


cdef double rndm_normal_blazin_fast(double mu, double sigma, int seed):
    cdef double rnd
    cdef double pi = 3.14159265358979323846
    srand48(seed)
    rnd = mu + sigma * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48())
    return rnd

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

    for i in range(num_sim*num_days+20):
        timestamps[i] = int(time.perf_counter() * 1e9)

    cdef double rnd
    cdef double pi = 3.14159265358979323846
    for x in range(num_sim*num_days):
        srand48(timestamps[x])
        randoms[x] = 0 + vola * sqrt(-2.0 * log(drand48())) * cos(2 * pi * drand48())



    print("num_sim: ", num_sim)
    print("num_days: ", (num_days - 1))
    print("iterator: ", iterator)
    print("vola: ", vola)
    print("lp: ", lp)
    print("len(close_list): ", len(close_list))
    print("len(returns): ", len(returns))

    # multiprocessing not very effective due lack of memory on CPU
    for y in prange(num_sim, nogil=True, num_threads=4):
        # 1 simulation
        for x in range(num_days - 1):
            if x == 0:
                sim_array[y][0] = lp * (1 + randoms[y * num_days + x])
            else:
                sim_array[y][x] = sim_array[y][x-1] * (1 + randoms[y * num_days + x])

    return convert_to_python(sim_array, num_sim, num_days) 


