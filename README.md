# Deep-Subspace-Clustering
Description pending...

## Sources used
* [X. Peng *et al*., [IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/275.pdf), [IEEE 2018](http://www.pengxi.me/wp-content/uploads/Papers/2018-TIP-StructAE.pdf)] - Deep Subspace Clustering with Sparsity Prior
  * [Author code](https://github.com/tonyabracadabra/Deep-Subspace-Clustering)
  * [Naive implementation](https://github.com/JasonJiaxiangLi/Deep-subspace-clustering-LDA-preprocess)
* [P. Ji *et al*., [NIPS 2017](https://papers.nips.cc/paper/6608-deep-subspace-clustering-networks.pdf)] - Deep Subspace Clustering Networks
  * [Author code](https://github.com/panji1990/Deep-subspace-clustering-networks)

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
If the last line fails due to denied permissions, point it to a directory that you have write permissions for. You can delete this directory afterwards:
```bash
python setup.py build --build-base=/full/path/to/temp install
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

Indicate that you'll be using Octave. If on linux, add this to your ```.bashrc```. If on windows, use ```SETX```:
```bash
export ENGINE_CHOICE=OCTAVE
```

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
octave:1> addpath(genpath("~/path/to/dir/usr/share/octave/packages/statistics-1.2.4"))
octave:2> savepath
```

Verify that kmeans works:
```octave
octave:3> kmeans(randn(10, 2), 2)
```

### Packages
Through ```pip``` or ```conda```, install the following:
* scikit-learn
* scikit-optimize
* munkres
* scipy
* matplotlib
* pympler
* pathlib
* mnist

### SSC using ADMM
Download code for [SSC using ADMM](http://vision.jhu.edu/code/). Unzip the contents into ```SSC_ADMM_v1.1```, located at the root of the repository.

### Datasets
Download and unzip any desired datasets. By default, the loading methods look for a ```data``` folder at the root of the repository.
* [Extended YaleB](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html) (cropped images)
* [Coil20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) (processed)
* [MNIST](http://yann.lecun.com/exdb/mnist/) will be automatically downloaded the first time ```load_MNIST``` is run
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) (python version)

Alternatively, you can just load the already preprocessed matrices from ```saved```.
