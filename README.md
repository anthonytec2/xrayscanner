# Python 3 x-ray simulator  
[Blog Post 1 X-Ray imaging background](https://abisulco.com/x-ray-imaging-pt1)  
[Blog Post 2 Speeding up Python code](https://abisulco.com/fast-python)  

The code in this repository was meant to simulate an imaging scenario described in this picture bellow. All the variables given are parameters for the imaging function. A matrix of linear attenuation coefficients is given representing the imaging object(headct.h5). The code that simulates the imaging scenario is x-ray-numba.py. In this script the imaging object is created, the energy at each detector pixel is calculated and then the detector image is saved. This code has been accelerated through the use of Numba which provided around 100x speed up over a pure python version. A script is given to x-ray-numba.py in order to calculate the energy pattern on a detector. A file will be saved called x-ray.png with the results once the script is run. 
Runtime 1.65s, code tested on Ubuntu 16.04, Numpy 1.15.0, Numba 0.37.0, h5py 2.7.1, Matplotlib 1.5.1, MKL 2018.1.163  

# CUDA Updated Version 

I have also developed a GPU optimized version of this code using Numba and CUDA. This file is named x-ray-cuda-numba.py and runs around 64x faster than the Numba version and ~6400x compared to pure Python. For the most part the results are the same between this version and the Numba version. One issue though is some of the GPU's floating point operations error propagation. Hence in a study I performed of a 300x300 pixel detector, 4 pixels had an unexpected difference from the Numba version. I have tried this code in the CUDA simulator but when porting to the GPU, multiple different little tweaks needed to be performed on the algorithm to make it work. Please see various comments on some of these tweaks. If you know how to fix any of these tweaks to work better please let me know.

```
pip3 install -r requirements.txt
python3 x-ray-numba.py
```

<img src = "profile/cube.png" width = "40%" >
