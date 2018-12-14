# Deep-Subspace-Clustering
Description pending...

## Installation
Download or ```git clone``` the repository.

### Prerequisites
* [Tensorflow](https://www.tensorflow.org/install/) - installed in a working Python environment such that ```import tensorflow``` throws no errors
* [Matlab](https://www.mathworks.com/help/install/ug/install-mathworks-software.html) - has an activated license, can be launched from desktop
* OR [Octave](https://www.gnu.org/software/octave/#install) - can be launched via ```octave``` in console

### M-files

#### Option 1: MATLAB API for Python
Requires a working and activated version of [MATLAB](https://www.mathworks.com/help/install/ug/install-mathworks-software.html).

Instructions taken from [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

Launch MATLAB and run:
```matlab
>>> matlabroot
```

Run the following on the command line, substituting the previous result into the first line. Make sure that ```python``` references a python version that has tensorflow installed:
```bash
cd matlabroot
cd extern/engines/python
python setup.py install
```
If installation fails due to denied permissions, [install/build in a non-default location](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html). If installing in a non-default fonder, don't forget to add that folder to your ```PYTHONPATH```.

Check the installation by running in Python:
```python
>>> import matlab.engine
>>> eng = matlab.engine.start_matlab()
>>> eng.isprime(37)
```

#### Option 2: Python to Octave bridge
A viable substitute for not having MATLAB. Requires a working version of [Octave](https://www.gnu.org/software/octave/#install).

Install oct2py. Make sure that ```pip``` is tied to a python version that has tensorflow installed:
```bash
pip install oct2py
```

Check the installation by running in Python:
```python
>>> from oct2py import octave
>>> octave.isprime(37)
```

Indicate that you'll be using Octave by changing the first non-import line of ```full_model.py``` from ```eng = start_matlab()``` to ```eng = start_octave()```.

### Packages
Through ```pip``` or ```conda```, install the following:
* sklearn
* scikit-optimize
* munkres
* scipy
* matplotlib

### SSC using ADMM
Download code for [SSC using ADMM](http://vision.jhu.edu/code/). Unzip the contents into ```SSC_ADMM_v1.1```, located at the root of the repository.

### Datasets
Download any desired datasets. By default, the loading methods look for a ```data``` folder at the root of the repository.
* [Extended YaleB](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html) (cropped images)
* [Coil20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) (processed)

Alternatively, you can just load the already preprocessed matrices from ```saved```.
