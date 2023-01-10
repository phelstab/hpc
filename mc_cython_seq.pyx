#cython: language_level=3
"""
    HPC part, monte carlo simulation
"""

import random
import math
import cython
from cpython cimport array
from cython cimport view

"""Draw random samples from mean loc and standard deviation scale"""
cpdef rndm_normal(mu, sigma):
    return random.gauss(mu, sigma)

"""
    Draw random samples from mean loc and standard deviation scale. With Box Muller transform
    https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    Inspiration -> https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
"""
# cpdef random_normal(mu, sigma):
#     U1 = random.uniform(-1, 1)
#     U2 = random.uniform(-1, 1)
#     W = U1 ** 2 + U2 ** 2
#     while W >= 1 or W == 0:
#         U1 = random.uniform(-1, 1)
#         U2 = random.uniform(-1, 1)
#         W = U1 ** 2 + U2 ** 2
#     mult = math.sqrt((-2 * math.log(W)) / W)
#     X1 = U1 * mult
#     X2 = U2 * mult
#     return mu + sigma * X1


cpdef double get_sum(series):
    cdef double _sum = 0.0
    cdef Py_ssize_t i
    for i in range(len(series)):
        _sum += series[i]
    return _sum

cimport cython

#https://en.wikipedia.org/wiki/Standard_deviation
# we need series in python rest in c    
cpdef double get_standard_deviation(double[:] series):
    # Calculate the mean of the series
    cdef double _sum = 0.0
    cdef Py_ssize_t i
    cdef signed int n = len(series)
    for i in range(n):
        _sum += series[i]
    cdef double _mean = _sum / n
    # Differences between each element and the mean
    cdef double x
    differences = [(x - _mean) ** 2 for x in series]
    # Variance by dividing the sum of the differences by the length of the series
    cdef double _sumdiff = get_sum(differences)
    cdef double variance = _sumdiff / n
    # Standard deviation, square root of the variance
    cdef double std_dev = variance ** 0.5
    return std_dev


def get_standard_deviation_old(series):
    # Calculate the mean of the series
    _sum = get_sum(series)
    _mean = _sum / len(series)
    # Differences between each element and the mean
    differences = [(x - _mean) ** 2 for x in series]
    # Variance by dividing the sum of the differences by the length of the series
    _sumdiff = get_sum(differences)
    variance = _sumdiff / len(series)
    # Standard deviation, square root of the variance
    std_dev = variance ** 0.5
    return std_dev


#Alt to: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
cpdef double[:] get_pct_change(double[:] series):
    cdef signed int n = len(series)
    cdef double pct_change
    cdef Py_ssize_t i
    cdef array.array pct_changes = array.array('d')
    for i in range(1, n):
        # Percentage change between the current element and the previous element
        pct_change = (series[i] - series[i-1]) / series[i-1]
        pct_changes.append(pct_change) 
    return pct_changes


"""
    Main monte-carlo part
"""

cpdef run(close_list, int num_sim, int num_days, double last_price):
    # convert python list to c array
    cdef array.array n_close_list = array.array('d', close_list)
    # prepare data for sim
    returns = get_pct_change(n_close_list)
    cdef double vola = get_standard_deviation(returns)
    
    
    cdef double price
    sim_array = [[]]
    cdef array.array price_series = array.array('d')
    for x in range(num_sim):
        price_series = array.array('d')
        count = 0        
        price = last_price * (1 + rndm_normal(0, vola))  
        price_series.append(price)

        while count < num_days - 1:
            price = price_series[count] * (1 + rndm_normal(0, vola))
            price_series.append(price)
            count += 1
        sim_array.append(price_series)

    return sim_array
