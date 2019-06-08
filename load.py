import img2matrix
import mnist
import numpy as np
import pickle
import time
from pathlib import Path

def load_YaleB(path="./data/CroppedYale"):
    print("\nLoading YaleB...")
    print("----------------")
    start_time = time.time()

    train, test, img_size = img2matrix.batch_load_YaleB(path, truncate_num=999, images_per_person=None)
    images = train[0]
    labels = train[1]

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels

# props to https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html
def load_Coil20(path="./data/coil-20-proc"):
    print("\nLoading Coil20...")
    print("----------------")
    start_time = time.time()

    path = Path(path)
    images = []
    labels = []

    for img in path.glob("obj*.png"):
        name = img.name
        # extract label from file name
        try:
            label = int(name[3:name.find('__')])
        except ValueError:
            continue
            
        images.append(img2matrix.read_image(img))
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels

def load_MNIST(path="./data/MNIST"):
    print("\nLoading MNIST...")
    print("----------------")
    start_time = time.time()

    path = Path(path)
    if(not path.exists()):
        path.mkdir()
    mnist.temporary_dir = lambda: str(path)

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels)).astype(np.int32)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels

def load_CIFAR10(path="./data/cifar-10-batches-py"):
    print("\nLoading CIFAR10...")
    print("----------------")
    start_time = time.time()

    path = Path(path)
    images = np.empty((0, 32*32*3), dtype=np.uint8)
    labels = np.empty((0), dtype=np.int32)

    for batch in path.glob("*_batch*"):
        with open(str(batch), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        images = np.concatenate((images, dict[b'data']))
        labels = np.concatenate((labels, dict[b'labels']))

    images = np.moveaxis(images.reshape((-1, 3, 32, 32)), 1, -1)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels

def split(data, fraction):
    """ Splits the given data into two proportional sets. Selection is random, but points are kept in their original order.
    
        PARAMETERS
        ---
        data [ndarray]:
            Data to split. The split is done along the 0th axis.
        
        fraction [double]:
            Number in range [0, 1]. Corresponds to the amount of data in output array A.
        
        RETURNS
        ---
        A [ndarray]:
            Contains N*fraction points.
        
        B [ndarray]:
            Contains N*(1-fraction) points.
        
    """
    assert 0 <= fraction <= 1, "Fraction must be between 0 and 1"
    include_idx = np.random.choice(data.shape[0], round(data.shape[0]*fraction), replace=False)
    mask = np.zeros(data.shape[0], dtype=bool)
    mask[include_idx] = True
    return data[mask], data[~mask]

def split_mult(data_arr, fraction):
    """ Splits multiple data arrays in tandem.
    
        PARAMETERS
        ---
        data [list of ndarray]:
            Data arrays to split.
        
        fraction [double]:
            Number in range [0, 1]. Corresponds to the amount of data in output arrays As.
        
        RETURNS
        ---
        As [list of ndarray]:
            Same order as given.
            Each contains N*fraction points.
        
        Bs [list ofndarray]:
            Each contains N*(1-fraction) points.
        
    """
    assert 0 <= fraction <= 1, "Fraction must be between 0 and 1"
    N = data_arr[0].shape[0]
    include_idx = np.random.choice(N, round(N*fraction), replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[include_idx] = True
    return [data[mask] for data in data_arr], [data[~mask] for data in data_arr]