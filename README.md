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

If installation fails due to denied permissions, [install/build in a non-default location](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html).

If installing in a non-default folder, don't forget to add that folder to your ```PYTHONPATH```. This will be reset on a reload, but can be appended to the end of ```.bashrc``` to execute on every shell start:
```bash
export PYTHONPATH=$PYTHONPATH:~/installdir/lib/python2.7/site-packages
```

If that somehow doesn't work, it's possible to modify Python's search path. Run this before attempting to ```import matlab```:
```python
import sys
sys.path.insert(0, "/full/path/to/installdir/lib/python2.7/site-packages")
```

Check the installation by running in Python:
```python
>>> import matlab.engine
>>> eng = matlab.engine.start_matlab()
>>> eng.isprime(37)
``` 

#### Option 2: Python to Octave bridge
A viable substitute for not having MATLAB. Requires a working version of [GNU Octave](https://www.gnu.org/software/octave/#install).

Install oct2py. Make sure that ```pip``` is tied to a python version that has tensorflow installed:
```bash
pip install oct2py
```

Check the installation by running in Python:
```python
>>> from oct2py import octave
>>> octave.isprime(37)
```

Indicate that you'll be using Octave by changing ```eng = start_matlab()``` to ```eng = start_octave()``` in ```full_model.py```.

If running ```kmeans``` in the Octave console says that it's undefined, you need to install the statistics package:
```bash
sudo apt-get install octave-statistics
```
Then, add the line ```pkg load statistics``` to ```SpectralClustering.m```

If you don't have access to ```sudo```, do the following instead. Make sure to substitute in the correct version number:
```bash
apt-get download octave-statistics
dpkg -x octave-statistics_1.2.4-1_all.deb dir
```
```octave
octave:1> addpath(genpath("~/dir/usr/share/octave/packages/statistics-1.2.4"))
octave:2> savepath
```

Verify that kmeans works:
```octave
octave:1> kmeans(randn(10, 2), 2)
```

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
