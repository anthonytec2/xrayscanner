import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import numpy as np
import math
import matplotlib.pyplot as plt  
import move_step
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x

@profile
def main():
    Nx = 208
    Ny = 256 
    Nz = 225
    Mx = 200
    My = 200
    D = 2
    h = 50
    H = h + Nz + 200
    orginOffset = np.array(
            [(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0], dtype='float32')
    ep = np.array([Nx / 2, Ny / 2, H], dtype = 'float32')
    muBone = 0.573
    muFat = 0.193
    orginOffset = np.array([(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 +(Ny / 2), 0],dtype='float32') 
    ep = np.array([Nx / 2, Ny / 2, H],dtype='float32')                                              
    f = h5py.File('headct.h5', 'r')
    headct=np.array(f.get('ct'))
    headct=np.transpose(headct)
    det=f.get('det')
    det=np.transpose(det)
    mu=np.zeros((Nx,Ny,Nz),dtype='float32')
    detector=np.zeros((Mx,My),dtype='float32')
    mu[np.nonzero(headct>0)]=((headct[np.nonzero(headct>0)]-0.3)/(0.7))*(muBone-muFat)+muFat
    del headct
    for z in range(0,Mx*My): 
        j=z%Mx
        i=int(z/Mx) 
        #print(str(i)+' '+str(j))
        pos=np.array([orginOffset[0]+i*D,orginOffset[1]+D*j, 0],dtype='float32')
        dir=np.array((ep-pos)/np.linalg.norm(ep-pos),dtype='float32')
        L=1
        #print('Dir'+str(dir))
        #print('Pos'+str(pos))
        while pos[2]< h+Nz:
            pos,dist=move_step.onemove_in_cube_true(pos,dir)
            if 0 <= pos[0] < Nx and 0<=pos[1]<Ny  and h<=pos[2] < h+Nz:
                L=L*np.exp(-1*mu[math.floor(pos[0]),math.floor(pos[1]),math.floor(pos[2]-h)]*dist)
        detector[i][j] = L;
    print(np.isclose(det,detector,rtol=.5).all())
    np.save('Detector.npy',detector)
    
'''
@profile
def onemove_in_cube_true(p0,v):   
    v[v==0]=1e-16
    htime=np.abs((np.floor(p0)-p0+(v>0))/v,dtype='float32')
    minLoc=htime.argmin()
    dist=htime[minLoc]
    htime=p0+dist*v
    htime[minLoc]=round(htime[minLoc])+np.spacing(abs(htime[minLoc]))*np.sign(v[minLoc])
    return htime, dist
'''

if __name__ == '__main__':
    main()
