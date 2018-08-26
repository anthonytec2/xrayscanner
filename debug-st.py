import h5py
import numpy as np
import numba
import timeit
import math
from numba import cuda
import pdb
import matplotlib.pyplot as plt
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


@cuda.jit(device=True, debug=True)
def onemove_in_cube_true_numba(p0, v):
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
    return htime, dist,minA


@cuda.jit(debug=True)
def main_loop(obj_dim, scene_info, orginOffset, ep, mu, detector,debuga):
    i, j = cuda.grid(2)
    if i < detector.shape[0] and j < detector.shape[1]:
        pos = cuda.local.array((3), dtype=numba.float32)
        direction = cuda.local.array((3), dtype=numba.float32)
        dol = cuda.local.array((3), dtype=numba.float32)
        pos[0] = orginOffset[0] + i * scene_info[0]
        pos[1] = orginOffset[1] + scene_info[0] * j
        pos[2] = 0
        norm = 0
        x = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bdx = cuda.blockDim.x
       
        for k in range(3):
            norm += math.pow(ep[k]-pos[k], 2)
        norm = math.sqrt(norm)
        for k in range(3):
            direction[k] = (ep[k]-pos[k])/norm
            if direction[k] == 0:
                direction[k] = 1e-16
        L = 0
        h_z = scene_info[1] + obj_dim[2]
        zz=0
        while pos[2] < h_z:
            if zz>5000:   
                break   
            p1, dist, minA = onemove_in_cube_true_numba(pos, direction)
            for k in range(3):
                pos[k]=p1[k]
            if 0 <= pos[0] < obj_dim[0] and 0 <= pos[1] < obj_dim[1] and scene_info[1] <= pos[2] < h_z:
                L += mu[int(math.floor(pos[0])), int(math.floor(pos[1])),
                        int(math.floor(pos[2] - scene_info[1]))] * dist
            zz+=1
        detector[i][j] = L

stream = cuda.stream()
h_detector = np.zeros((Mx, My), dtype=np.float32)
d_detector = cuda.to_device(h_detector, stream)
d_mu = cuda.to_device(mu, stream)
h_obj_dim = np.array([Nx, Ny, Nz])
d_obj_dim = cuda.to_device(h_obj_dim, stream)
h_scene_info = np.array([D, h])
d_scene_info = cuda.to_device(h_scene_info, stream)
d_ep = cuda.to_device(ep, stream)
d_orginOffset = cuda.to_device(orginOffset, stream)
h_debug=np.zeros((5000,3), dtype=np.float32)
d_debug=cuda.to_device(h_debug, stream)
stream.synchronize()
main_loop[(20, 20), (8, 8), stream](d_obj_dim, d_scene_info, d_orginOffset, d_ep, d_mu, d_detector, d_debug)
stream.synchronize()
res = d_detector.copy_to_host()
res2= d_debug.copy_to_host()
det2 = np.exp(res * -10, dtype=np.float64)

plt.imshow(np.log(det2))
plt.savefig('image.png')
#plt.show()
