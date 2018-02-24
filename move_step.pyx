import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
# cython: np_pythran=True
@cython.boundscheck(False) 
@cython.wraparound(False) 
def onemove_in_cube_true(np.ndarray[DTYPE_t] p0, np.ndarray[DTYPE_t] v): 
    v[v==0]=1e-16
    cdef np.ndarray[DTYPE_t] htime=np.abs((np.floor(p0)-p0+(v>0))/v)
    cdef int minLoc=htime.argmin()
    cdef float dist=htime[minLoc]
    htime=p0+dist*v
    htime[minLoc]=round(htime[minLoc])+np.spacing(1e10*abs(htime[minLoc]))*np.sign(v[minLoc])
    return htime,dist
