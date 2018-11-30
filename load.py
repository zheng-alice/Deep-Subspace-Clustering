import img2matrix
import time
from pathlib import Path
import numpy as np

def load_YaleB(path="./data/CroppedYale"):
    print("\nLoading YaleB...")
    print("----------------")
    start_time = time.time()

    train, test, img_size = img2matrix.batch_convert_YaleB(path, truncate_num=38, images_per_person=None)
    images = train[0]
    labels = train[1]

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels

# props to https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html
def load_Coil20(path="./data/coil-20-proc"):
    print("\nLoading Coil20...")
    print("----------------")
    start_time = time.time()

    path = Path("./data/coil-20-proc")
    images = []
    labels = []

    for img in path.glob("obj*.png"):
        name = img.name
        # extract label from file name
        try:
            label = int(name[3:name.find('__')])
        except ValueError:
            continue
            
        images.append(img2matrix.single_img2dsift(img))
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images, labels