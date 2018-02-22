import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
def onemove_in_cube_true(np.ndarray[DTYPE_t] p0,np.ndarray[DTYPE_t] v): 
    cdef np.ndarray[DTYPE_t] htime=np.abs((np.floor(p0)-p0+(v>0))/v)
    cdef int minLoc=htime.argmin()
    cdef float dist=htime[minLoc]
    htime=p0+dist*v
    htime[minLoc]=round(htime[minLoc])+np.spacing(abs(htime[minLoc]))*np.sign(v[minLoc])
    return htime, dist
