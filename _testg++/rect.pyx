#cython: language_level=3
# distutils: language = c++

from Rectangle cimport Rectangle
from random_gauss cimport RandomGauss
# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyRectangle:
    cdef Rectangle c_rect  # Hold a C++ instance which we're wrapping

    def __init__(self, int x0, int y0, int x1, int y1):
        self.c_rect = Rectangle(x0, y0, x1, y1)

    def get_area(self):
        return self.c_rect.getArea()

    def get_size(self):
        cdef int width, height
        self.c_rect.getSize(&width, &height)
        return width, height

    def move(self, dx, dy):
        self.c_rect.move(dx, dy)

# random_gauss.pyx
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


def main():
    # print a rectangle
    cdef PyRectangle rect = PyRectangle(0, 0, 10, 10)
    print(rect.get_size())
    # define a random number generator
    cdef PyRandomGauss gauss = PyRandomGauss(0, 1)
    print(gauss.next())
    