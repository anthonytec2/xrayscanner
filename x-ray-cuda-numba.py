'''
Code developed by Anthony Bisulco
August 28th, 2018
This is a script to run an imaging scenario on a given imaging volume
with an arbitrary sized detector. Currently, the file that loads is
a brain scan which is imaged. This code uses Numba acceleration in order
to obtain reasonable imaging times. Right now the ray tracing is off by 4 pixels
in a 300x300 image, possibly due to GPU IEEE floating point errors.
'''
import h5py
import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt


@cuda.jit(device=True)
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
    htime = cuda.local.array((3), dtype=numba.float32)
    for i in range(3):
        if v[i] > 0:
            htime[i] = abs((math.floor(p0[i]) - p0[i] + 1) / v[i])
        else:
            htime[i] = abs((math.floor(p0[i]) - p0[i] - 1e-4) / v[i])
    minA = 0
    minV = htime[0]
    for i in range(1, 3):
        if minV > htime[i]:
            minA = i
            minV = htime[i]
    dist = htime[minA]

    for i in range(3):
        htime[i] = p0[i] + dist * v[i]
    if v[minA] < 0:
        htime[minA] = round(htime[minA]) + 1.5e-4 * -1
    else:
        htime[minA] = round(htime[minA]) + 1.5e-4
    return htime, dist


@cuda.jit
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

    i, j = cuda.grid(2)
    if i < detector.shape[0] and j < detector.shape[1]:
        pos = cuda.local.array((3), dtype=numba.float32)
        direction = cuda.local.array((3), dtype=numba.float32)
        pos[0] = h_image_params[8] + h_image_params[3] * numba.float32(i)
        pos[1] = h_image_params[9] + h_image_params[3] * numba.float32(j)
        pos[2] = 0
        norm = 0
        for k in range(3):
            norm += math.pow(h_image_params[5+k]-pos[k], 2)
        norm = math.sqrt(norm)
        for k in range(3):
            direction[k] = (h_image_params[5+k]-pos[k])/norm
            if direction[k] == 0:
                direction[k] = 1e-16
        L = 0
        h_z = h_image_params[4] + h_image_params[2]
        while pos[2] < h_z:
            if -.0001 < pos[0] < .0001:
                pos[0] += .0001
            if -.0001 < pos[1] < .0001:
                pos[1] += .0001
            p1, dist = onemove_in_cube_true_numba(pos, direction)
            for k in range(3):
                pos[k] = p1[k]
            if 0 <= pos[0] < h_image_params[0] and 0 <= pos[1] < h_image_params[1] and h_image_params[4] <= pos[2] < h_z:
                L += mu[int(math.floor(pos[0])), int(math.floor(pos[1])),
                        int(math.floor(pos[2] - h_image_params[4]))] * dist
        detector[i][j] = L


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
stream = cuda.stream()
h_image_params = np.array([Nx, Ny, Nz, D, h, ep[0], ep[1], ep[2],
                           orginOffset[0], orginOffset[1], orginOffset[2]], dtype=np.float32)
d_detector = cuda.device_array((Mx, My), dtype=np.float32, stream=stream)
d_mu = cuda.to_device(mu, stream)
d_image_params = cuda.to_device(h_image_params, stream)
stream.synchronize()
main_loop[(int(np.ceil(Mx/25)), int(np.ceil(My/25))),
          (25, 25), stream](h_image_params, d_mu, d_detector)
stream.synchronize()
res = d_detector.copy_to_host()
det2 = np.exp(res * -10, dtype=np.float64)
plt.imshow(np.log(det2))
plt.savefig('image.png')
print('Done')
