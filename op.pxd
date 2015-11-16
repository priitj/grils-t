# declarations for cimport

cdef class FastDistMatrix:
    cdef double *table
    cdef int nitems

cdef class FastRCL(object):
    cdef int *sol_idx_m
    cdef int *cl_idx_m
    cdef double *cl_hval_m
    cdef int *cl_pos_m
    cdef double *cl_cost_m
    cdef int *rcl_idx_m
    cdef int ncand
    cdef object mkcand
    cdef object items

cdef inline double travel_cost(double *table, int nitems, int i, int j)

cdef double cost_delta_noshift(double *tc, int *sol_idx, int ik,
  int position, int maxpos, int nitems)

