
cdef extern from "random_gauss.cpp":
    pass

cdef extern from "random_gauss.h" namespace "gauss":
    cdef cppclass RandomGauss:
        RandomGauss(double mean, double stddev)
        double next()


cdef extern from "random_gauss.h":
    cdef cppclass RandomGauss:
        long long get_current_time_ns()
