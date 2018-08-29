'''
Code developed by Anthony Bisulco
August 28th, 2018
This is a script to run an imaging scenario on a given imaging volume
with an arbitrary sized detector. Currently, the file that loads is
a brain scan which is imaged. This code uses Numba acceleration in order
to obtain reasonable imaging times. Right now the ray tracing is off by 4 pixels
in a 300x300 image, possibly due to GPU IEEE floating point errors.
'''
import h5py  # library used for data import
import numpy as np  # library used for data manipulation
import numba  # library used to speed up code
from numba import cuda  # helper cuda for cuda functions
import math  # math library to perform in kernel
import matplotlib.pyplot as plt  # plotting library


@cuda.jit(device=True)  # CUDA Device function must be called from GPU kernel
def onemove_in_cube_true_numba(p0, v):
    '''
    This is a function that moves from a given position p0 in direction v to another cube in a 1x1x1mm setup
    Args:
        p0: np.array 1x3 start position (X,Y,Z)
        v: np.array 1x3 normalized(1) direction vector (X,Y,Z)

    Returns:
        htime: np.array 1x3 next cube position (X,Y,Z)
        dist: float distance to the next cube position

    '''
    htime = cuda.local.array((3), dtype=numba.float32) #Create local CUDA array to each thread to store next position
    for i in range(3): #loop through dim to calc time to next pos
        if v[i] > 0: #if positive need to add 1 to reverse negative result
            htime[i] = abs((math.floor(p0[i]) - p0[i] + 1) / v[i])
        else: #if negative, perform calc as usual, *CUDA requires the 1e-4 for some reason, code will not return without possibly floating pt issues
            htime[i] = abs((math.floor(p0[i]) - p0[i] - 1e-4) / v[i])
    min_loc = 0 # min location in array
    min_time = htime[0] # min value in array
    for i in range(1, 3): #loop through elem in time array
        if min_time > htime[i]: # check if smaller val
            min_loc = i # reset min loc
            min_time = htime[i] # set min val#

    for i in range(3): # loop through dim and calc next post
        htime[i] = p0[i] + min_time * v[i]

    if v[min_loc] < 0: # add incremental amount for next iter, to no get stuck in endless loop
        htime[min_loc] = round(htime[min_loc]) + 1.5e-4 * -1
    else:
        htime[min_loc] = round(htime[min_loc]) + 1.5e-4
    return htime, min_time # return next position and min distance/time

@cuda.jit # GPU Kernel must be executed from host
def main_loop(h_image_params, mu, detector):
    '''
    Ray tracing from end point to all pixels, calculates energy at every pixels
    Args:
        h_image_params:
            Nx: uint imaging volume length in x direction
            Ny: uint imaging volume length in y direction
            Nz: uint imaging volume length in z direction
            Mx: uint number of pixels in x direction
            My: uint number of pixels in y direction
            D: uint pixel length
            h: uint distance from detector to bottom of imaging volume
            orginOffset: np.array 1x2 offset origin for detector position start (X,Y)
            ep: np.array 1x3 location of the x-ray source (X,Y,Z)
        mu: np.array 1x3 normalized linear attenuation coefficient matrix (Nx,Ny,Nz)
        detector: np.array 1x3 next cube position (X,Y,Z)
    '''

    i, j = cuda.grid(2) # Create thread indices i, j
    if i < detector.shape[0] and j < detector.shape[1]: # check thread indices within detector, o.w. return
        pos = cuda.local.array((3), dtype=numba.float32) # create local thread storage for current ray position
        direction = cuda.local.array((3), dtype=numba.float32) # create local thread storage for current direction
        pos[0] = h_image_params[8] + h_image_params[3] * numba.float32(i) # X Calc ray start pos with origin offset and x pixel offset
        pos[1] = h_image_params[9] + h_image_params[3] * numba.float32(j) # Y Calc ray start pos with origin offset and y pixel offset
        pos[2] = 0 # start ray at detector Z=0
        norm = 0 # Calculate normalization for direction vector
        for k in range(3): # loop through dim
            norm += math.pow(h_image_params[5+k]-pos[k], 2) # calc squared diff bettween source loc to current pos
        norm = math.sqrt(norm) # take squareroot for normalization factor
        for k in range(3): # loop through direction dim
            direction[k] = (h_image_params[5+k]-pos[k])/norm # direction = source loc -position over norm
            if direction[k] == 0: # Dont want any divide by 0 errors, therefore set to small val
                direction[k] = 1e-16
        L = 0 # Start energy at source location
        h_z = h_image_params[4] + h_image_params[2] # highest Z location 
        while pos[2] < h_z: # loop until the ray exits the highest Z location, after that ray doesent attenuate
            if -.0001 < pos[0] < .0001: # CUDA needs this, tested in sim without algo works, need some incremental update around 0 to not get stuck
                pos[0] += .0001
            if -.0001 < pos[1] < .0001: # CUDA needs this, tested in sim without algo works, need some incremental update around 0 to not get stuck
                pos[1] += .0001
            p1, dist = onemove_in_cube_true_numba(pos, direction) # move from current pos in direction to new cube in mu grid
            for k in range(3): # need to copy over data for some reason in Numba or else array doesent get updated
                pos[k] = p1[k]
            if 0 <= pos[0] < h_image_params[0] and 0 <= pos[1] < h_image_params[1] and h_image_params[4] <= pos[2] < h_z: # if in imaging volume
                # calculate energy using mu
                L += mu[int(math.floor(pos[0])), int(math.floor(pos[1])),
                        int(math.floor(pos[2] - h_image_params[4]))] * dist
        detector[i][j] = L # detector pixel location equals lasting energy

def main():
    # HDF5 file containing Headct array of linear attenuation
    # coefficient(Nx,Ny,Nz)
    f = h5py.File('headct.h5', 'r')
    headct = np.array(f.get('ct'))
    headct = np.transpose(headct)  # linear attenuation coefficient matrix
    Nx = np.size(headct, 0)  # Imaging x dimension length in mm
    Ny = np.size(headct, 1)  # Imaging y dimension length in mm
    Nz = np.size(headct, 2)  # Imaging z dimension length in mm
    Mx = 128  # Number of pixels in x direction
    My = 128  # Number of pixels in y direction
    h = 2  # distance(Z) bettween bottom of imaging volume and detector
    H = h + Nz + 600  # distance(Z) bettween detector and x-ray source
    dx = (H * Nx) / ((H - Nz - h) * Mx)  # distance x direction for each pixel
    dy = (H * Ny) / ((H - Nz - h) * My)  # distance y direction for each pixel
    D = max(dx, dy)  # Size of each pixel in mm
    muBone = 0.573  # linear attenuation coefficient bone cm^-1
    muFat = 0.193  # linear attenuation coefficient fat cm^-1
    # offset from origin to detector start (X,Y,Z)
    orginOffset = np.array(
        [(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0], dtype=np.float32)
    # location of x-ray soruce
    ep = np.array([Nx / 2, Ny / 2, H], dtype=np.float32)
    # offset from origin to detector start
    orginOffset = np.array(
        [(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0], dtype=np.float32)
    # (Nx,Ny,Nz) linear attenuation coefficient matrix
    mu = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    mu[np.nonzero(headct > 0)] = ((headct[np.nonzero(headct > 0)] - 0.3) / (0.7)) * \
        (muBone - muFat) + muFat  # Normalization of givens mus of linear attenuation matrix
    stream = cuda.stream() # Create CUDA stream for async transfer to GPU
    h_image_params = np.array([Nx, Ny, Nz, D, h, ep[0], ep[1], ep[2],
                            orginOffset[0], orginOffset[1], orginOffset[2]], dtype=np.float32) # Create an array withing imaging params 
    d_detector = cuda.device_array((Mx, My), dtype=np.float32, stream=stream) # Create a device error on GPU for detector results
    d_mu = cuda.to_device(mu, stream) # Transfer the linear attenuation coefficients to GPU
    d_image_params = cuda.to_device(h_image_params, stream) # Transfer imaging params to GPU
    stream.synchronize() # Force all memory transfer to occur before starting func
    main_loop[(int(np.ceil(Mx/25)), int(np.ceil(My/25))),
            (25, 25), stream](h_image_params, d_mu, d_detector) # start main loop function and create x-ray image
    stream.synchronize() # Force main loop to finish before memory transferback
    res = d_detector.copy_to_host() # Transfer detector image to host
    det2 = np.exp(res * -10, dtype=np.float64) # Perform necessary rescaling 
    plt.imshow(np.log(det2)) # Plot result
    plt.title('Detector Log Image Python')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.colorbar()
    plt.savefig('x-ray.png') # Save Figure
    print('Done') # print finshed

if __name__ == '__main__':
    main()
