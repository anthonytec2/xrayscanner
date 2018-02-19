import numpy as np
import h5py   
import math
import matplotlib.pyplot as plt  
from joblib import Parallel, delayed
import multiprocessing

def onemove_in_cube_true(p0,v):
    htime=np.abs((np.floor(p0)-p0+(v>0))/v)
    minLoc=htime.argmin()
    dist=htime[minLoc]
    htime=p0+dist*v
    htime[minLoc]=np.round(htime[minLoc])+np.spacing(np.abs(np.float64(htime[minLoc])))*np.sign(v[minLoc])
    return htime, dist

def onemove_in_cube_true_ut(p0,v):
    htime=np.abs((np.floor(p0)-p0+(v>0))/v)
    minLoc=htime.argmin()
    dist=htime[minLoc]
    htime=p0+dist*v
    htime[minLoc]=np.round(htime[minLoc])+np.spacing(np.abs(htime[minLoc]))*np.sign(v[minLoc])
    return htime, dist

def sing_pixel(z,orginOffset,ep,D,h,Nx,Ny,Nz,mu,Mx):
    j=z%Mx
    i=int(z/Mx)
    pos=np.array([orginOffset[0]*i+D,orginOffset[1]+D*j, 0])
    dir=np.array((ep-pos)/np.linalg.norm(ep-pos))
    L=1
    while pos[2]< h+Nz:
        pos,dist=onemove_in_cube_true_ut(pos,dir)
        if 0 <= pos[0] < Nx and 0<=pos[1]<Ny  and h<=pos[2] < h+Nz:
            L=L*np.exp(-1*mu[math.floor(pos[0]),math.floor(pos[1]),math.floor(pos[2]-h)]*dist)
    return L

def main():
    Nx = 208
    Ny = 256
    Nz = 225
    Mx = 128
    My = 128
    D = 2
    h = 50
    H = h + Nz + 200
    orginOffset = np.array(
            [(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0], dtype='float64')
    ep = np.array([Nx / 2, Ny / 2, H], dtype = 'float64')
    muBone = 0.573
    muFat = 0.193
    orginOffset = np.array([(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0]) 
    ep = np.array([Nx / 2, Ny / 2, H])                                              
    f = h5py.File('headct.h5', 'r')
    headct=np.array(f.get('ct'))
    mu=np.zeros((Nx,Ny,Nz))
    mu[np.nonzero(mu>0)]=((headct[np.nonzero(mu>0)]-0.3)/(0.7))*(muBone-muFat)+muFat
    del headct
    detector=np.zeros((Mx,My))
    pos=np.zeros(3)
    dir=np.zeros(3)
    num_cores = multiprocessing.cpu_count()
    inputs=range(0,Mx*My)
    results = Parallel(n_jobs=num_cores)(delayed(sing_pixel)(i,orginOffset,ep,D,h,Nx,Ny,Nz,mu,Mx) for i in inputs)

'''
    for z in range(0,Mx*My): 
        j=z%Mx
        i=int(z/Mx) 
        #print(str(i)+' '+str(j))
        pos=np.array([orginOffset[0]*i+D,orginOffset[1]+D*j, 0])
        dir=np.array((ep-pos)/np.linalg.norm(ep-pos))
        L=1
        #print('Dir'+str(dir))
        while pos[2]< h+Nz:
            pos,dist=onemove_in_cube_true_ut(pos,dir)
            if 0 <= pos[0] < Nx and 0<=pos[1]<Ny  and h<=pos[2] < h+Nz:
                L=L*np.exp(-1*mu[math.floor(pos[0]),math.floor(pos[1]),math.floor(pos[2]-h)]*dist)
        detector[i][j] = L;   
'''

if __name__ == '__main__':
    main()
