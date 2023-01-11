
#cython: language_level=3

import random
import math
from random import random as rand

import cython
from cpython cimport array
from cython cimport view
from cython.parallel import  prange, threadid
from libc.stdlib cimport malloc, free
import time

"""Draw random samples from mean loc and standard deviation scale"""
# pure in c (very fast) > pure python with math and random function (fast) > random.gauss (slow) > (very slow)  
cdef rndm_normal(mu, sigma):
    cdef double rnd = random.gauss(mu, sigma)
    return rnd

def rndm_normal_new(mu, sigma):
    rnd = mu + sigma * math.sqrt(-2.0 * math.log(random.random())) * math.cos(2 * math.pi * random.random())
    return rnd


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


cdef double rndm_normal_blazin_fast(double mu, double sigma):
    cdef double rnd
    cdef double pi = 3.14159265358979323846
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
    #print("ignore1: ", len(series))
    cdef int threads = 8
    cdef signed int n = 5790
    with nogil:
        for i in prange(n, num_threads=threads):
            _sum += series[i]
    return _sum

from libc.stdlib cimport malloc, free
@cython.boundscheck(False)
cdef double[:] get_differences_from_mean(double[:] series, double _mean):
    cdef signed int n = 5790
    cdef double* result = <double*>malloc(n * sizeof(double))
    cdef Py_ssize_t g
    cdef int threads = 8
    with nogil:
        for g in prange(n, num_threads=threads):
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
        # Percentage change between the current element and the previous element
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
    #cdef double* aligned_randoms = <double*>((<uintptr_t>randoms + 15) & ~0xF)
    cdef double** sim_array = <double**>malloc((num_sim * sizeof(double*)) + 15)
    #cdef double** aligned_sim_array = <double**>((<uintptr_t>sim_array + 15) & ~0xF)
    #cdef aligned_sim_array = <double*>((<uintptr_t>aligned_sim_array + 15) & ~0xF)

    # malloc sim array size of days for each sim
    for x in range(num_sim):
        sim_array[x] = <double*>malloc((num_days * sizeof(double)) + 15)
        #aligned_sim_array[x] = <double*>((<uintptr_t>aligned_sim_array[x] + 15) & ~0xF)
    
    # create randoms for each sim and day and store in 1 dimension
    for x in range(num_sim*num_days):
        randoms[x] = rndm_normal_blazin_fast(0, vola)
    
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

    print("num_sim: ", num_sim)
    print("num_days: ", (num_days - 1) // 2) # check, must be even
    print("iterator: ", iterator)
    print("vola: ", vola)
    print("lp: ", lp)
    print("len(close_list): ", len(close_list))
    print("len(returns): ", len(returns))

    #cdef __m128d a, b, c, d
    # with nogil:
    #     one = _mm_loadu_pd( &p[0] )
    for y in range(num_sim):
        # 1 simulation
        for x in range(num_days - 1):
            if x == 0:
                sim_array[y][0] = lp * (1 + randoms[y * num_days + x])
            else:
                sim_array[y][x] = sim_array[y][x-1] * (1 + randoms[y * num_days + x])
                    # a = _mm_load_pd(&sim_array[y][x-1])
                    # b = _mm_load_pd(&randoms[y * num_days + x])
                    # c = _mm_add_pd(a, b)
                    # d = _mm_mul_pd(a, c)
                    # _mm_store_pd(&sim_array[y][x], d)
                    #_mm_store_pd( &out[2*x], mtmp )
                    # store the result in the output array
                    #sim_array[y][x] = out[0]
                    #sim_array[y][x+1] = out[1]

    #free(randoms)
    return convert_to_python(sim_array, num_sim, num_days) 



cdef extern from "emmintrin.h":  # in this example, we use SSE2
    # Two things happen here:
    # - this definition tells Cython that, at the abstraction level of the Cython language, __m128d "behaves like a double"
    # - at the C level, the "cdef extern from" (above) makes the generated C code look up the exact definition from the original header
    #
    ctypedef double __m128d

    # Declare any needed extern functions here; consult $(locate emmintrin.h) and SSE assembly documentation.
    #
    # For example, to pack an (unaligned) double pair, to perform addition and multiplication (on packed pairs),
    # and to unpack the result, one would need the following:
    #
    __m128d _mm_loadu_pd (double *__P) nogil  # (__P[0], __P[1]) are the original pair of doubles
    __m128d _mm_add_pd (__m128d __A, __m128d __B) nogil
    __m128d _mm_mul_pd (__m128d __A, __m128d __B) nogil
    void _mm_store_pd (double *__P, __m128d __A) nogil  # result written to (__P[0], __P[1])


