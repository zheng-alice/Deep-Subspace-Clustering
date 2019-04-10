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

    train, test, img_size = img2matrix.batch_load_YaleB(path, truncate_num=38, images_per_person=None)
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