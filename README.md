# Python 3 x-ray simulator
The code in this repository was meant to simulate an imaging scenario described in this picture given. All the variables given are parameters for the imaging function. A matrix of linear attenuation coefficients is given representing the imaging object(headct.h5). The code that simulates the imaging scenario is bio.py. In this script the imaging object is created, the energy at each detector pixel is calculated and then saved to a Numpy save file. This code has been accelerated through the use of Numba which provided around 100x speed up over a pure python version. A script is given to bio.py in order to calculate the energy pattern on a detector. A file will be saved called plot.png with the results once the script is run. 
Runtime 1.65s, code tested on Ubuntu 16.04, Numpy 1.15.0, Numba 0.37.0, h5py 2.7.1, Matplotlib 1.5.1, MKL 2018.1.163

```
pip3 install -r requirements.txt
python3 bio.py
```

<img src = "profile/cube.png" width = "40%" >
