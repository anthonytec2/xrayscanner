'''
Code developed by Anthony Bisulco
February 26th, 2018
This is a script to run an imaging scenario on a given imaging volume
with an arbitrary sized detector. Currently, the file that loads is
a brain scan which is imaged. This code uses Numba acceleration in order
to obtain reasonable imaging times.
'''

import h5py
import numpy as np
import math
import numba
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import figure, colorbar, savefig, title, xlabel, ylabel, imshow


@numba.jit(nopython=True, nogil=True, cache=True)
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

    # find distance vector to new position
    htime = np.abs((np.floor(p0) - p0 + (v > 0)) / v)
    minLoc = np.argmin(htime)  # find min distance location in htime
    dist = htime[minLoc]  # find min distance
    htime = p0 + dist * v  # calculate new position estimate htime
    # Need this for rounding to next position
    htime[minLoc] = round(htime[minLoc]) + \
        np.spacing(abs(htime[minLoc])) * np.sign(v[minLoc])
    return htime, dist


@numba.jit(nopython=True, nogil=True, cache=True)
def main_loop(Nx, Ny, Nz, Mx, My, D, h, orginOffset, ep, mu):
    '''
    Ray tracing from end point to all pixels, calculates energy at every pixels
    Args:
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
    Returns:
        detector: np.array 1x3 next cube position (X,Y,Z)
    '''
    detector = np.zeros(
        (Mx, My), dtype=np.float32)  # detector Mx x pixels and My y pixels
    for z in range(0, Mx * My):  # loop for all pixels
        j = z % Mx  # y direction pixel
        i = int(z / Mx)  # x direction pixel
        pos = np.array([orginOffset[0] + i * D, orginOffset[1] + D * j,
                        0], dtype=np.float32)  # pixel location
        # normalized direction vector to source
        dir = ((ep - pos) / np.linalg.norm(ep - pos)).astype(np.float32)
        # need this for floating point division errors in Numba
        dir[dir == 0] = 1e-16
        L = 0  # initial energy
        h_z = h + Nz
        while pos[2] < h_z:  # loop until the end of imaging volume
            pos, dist = onemove_in_cube_true_numba(
                pos, dir)  # move to next cube
            if 0 <= pos[0] < Nx and 0 <= pos[1] < Ny and h <= pos[2] < h_z:  # if in imaging volume
                # calculate energy using mu
                L += mu[math.floor(pos[0]), math.floor(pos[1]),
                        math.floor(pos[2] - h)] * dist
        detector[i][j] = L  # detector pixel location equals lasting energy
    return detector


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
        (muBone - muFat) + \
        muFat  # Normalization of givens mus of linear attenuation matrix
    detA = main_loop(Nx, Ny, Nz, Mx, My, D, h, orginOffset, ep, mu)
    detector = np.exp(detA * -10, dtype=np.float64)
    fig = figure()
    imshow(np.log(detector), extent=[0, Mx, 0, My], cmap='viridis')
    title('Detector Log Image Python')
    xlabel('X Pixel')
    ylabel('Y Pixel')
    colorbar()
    savefig('plot.png')


if __name__ == '__main__':
    main()
