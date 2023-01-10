from cython.parallel import prange
from libc.stdlib cimport malloc, free, abort

cpdef c_array_f_multi():
    #
    cdef int i
    cdef int n = 30
    cdef int sum = 0

    for i in prange(n, nogil=True):
        sum += i

    print(sum)