
#cython: language_level=3

import random
import math
import cython
from cpython cimport array
from cython cimport view
from cython.parallel import  prange, threadid
from libc.stdlib cimport malloc, free

"""Draw random samples from mean loc and standard deviation scale"""
cdef rndm_normal(mu, sigma):
    cdef double rnd = random.gauss(mu, sigma)
    return rnd

#https://stackoverflow.com/a/32746954
#get generate random double from seed
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

#get time from c extern
cdef extern from "time.h":
    long int time(int)

#get log from c extern
cdef extern from "math.h":
    double log(double x)

#get sqrt from c extern
cdef extern from "math.h":
    double sqrt(double x)

# cdef get_rand():
#     srand48(time(0))
#     return drand48()

# cdef get_log(double x):
#     cdef double _x = log(x)
#     return _x

# cdef get_sqrt(double x):
#     cdef double _x = sqrt(x)
#     return _x


# cdef random_normal(double mu, double sigma):
#     cdef double U1 = get_rand()
#     cdef double U2 = get_rand()
#     cdef double W = U1 ** 2 + U2 ** 2
#     while W >= 1 or W == 0:
#         U1 = get_rand()
#         U2 = get_rand()
#         W = U1 ** 2 + U2 ** 2
#     cdef double mult = get_sqrt((-2 * get_log(W)) / W)
#     cdef double X1 = U1 * mult
#     cdef double X2 = U2 * mult
#     return mu + sigma * X1

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
    #print("ignore3: ", len(series))
    #cdef double pct_change
    cdef Py_ssize_t i
    #cdef array.array pct_changes = array.array('d')
    cdef double* pct_changes = <double*>malloc(n * sizeof(double))
    for i in range(1, n):
        # Percentage change between the current element and the previous element
        pct_changes[i] = (series[i] - series[i-1]) / series[i-1]
        #pct_changes.append(pct_change) 
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

@cython.boundscheck(False)
cpdef run(close_list, int num_sim, int num_days, double last_price):
    # convert python list to c array
    cdef array.array n_close_list = array.array('d', close_list)
    # prepare data for sim
    returns = get_pct_change(n_close_list)
    cdef double vola = get_standard_deviation(returns)
    cdef double lp = last_price
    cdef signed int count = 0
    cdef double rnd 
    cdef double** sim_array =  <double**>malloc(num_sim * sizeof(double))
    cdef int x
    cdef int y
    cdef int static_iterator
    cdef int s_counter
    cdef signed int iterator = num_sim*num_days
    cdef double* randoms = <double*>malloc(iterator * sizeof(double))
    cdef int* localCounts = <int*>malloc(iterator * sizeof(int))
    cdef int tid
    
    # malloc sim array size of days for each sim
    for x in range(num_sim):
        sim_array[x] = <double*>malloc(num_days * sizeof(double))
    
    # create randoms for each sim and day and store in 1 dimension
    for x in range(num_sim*num_days):
        randoms[x] = rndm_normal(0, vola)
    
    # for each simulation
    # num_sim: 1000
    # num_days: 5791
    # x = 0 to 5790
    # y = 0 to 1000
    # c = the random counter from 0 to 5790*1000
    # randoms[] containing 5790*1000 randoms for every price
    for y in prange(num_sim, nogil=True, schedule='dynamic', num_threads=8):
        # 1 simulation
        for x in range(num_days - 1):
            if x == 0:
                sim_array[y][0] = lp * (1 + randoms[y * num_days + x])
            else:
                sim_array[y][x] = sim_array[y][x-1] * (1 + randoms[y * num_days + x])
        with gil:
            tid = threadid()
            print("thread {:} finished: ".format(tid))
        

    return convert_to_python(sim_array, num_sim, num_days) 
