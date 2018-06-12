# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------

#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
#cython: cdivision=True
import numpy
cimport numpy
from libcpp.vector cimport vector


cdef extern from 'cchips.h' namespace 'chips':
    vector[vector[float]] cgenerate(int width, int height, int chipsize, vector[vector[float]]& boxes, int num_boxes, int stride)

def generate(numpy.ndarray[float,ndim=2] boxes, int width, int height, int chipsize, int stride):
    res = cgenerate(width, height, chipsize, boxes, boxes.shape[0], stride)
    return res


