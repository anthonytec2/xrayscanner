import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import numpy as np
import matplotlib.pyplot as plt  

f = h5py.File('headct.h5', 'r')
det=f.get('det')
det=np.transpose(det)
detector=np.load('Detector.npy')
plt.figure(1)
plt.imshow(det, extent=[0, 1, 0, 1])
plt.title('Detector Matlab')
plt.figure(2)
plt.imshow(detector, extent=[0, 1, 0, 1])
plt.title('Detector Python')
plt.figure(3)
plt.imshow(np.log(det), extent=[0, 1, 0, 1])
plt.title('Detector Log Matlab')
plt.figure(4)
plt.imshow(np.log(detector), extent=[0, 1, 0, 1])
plt.title('Detector Log Python')
plt.show()
