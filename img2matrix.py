# Third-party libraries
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import dsift
from scipy import misc


def single_img2matrix(pgm_dir, k=4):
    # k is the pooling size
    img = Image.open(pgm_dir)
    # img.show()
    data_mat = np.array(img)
    # We now do pooling and reshape
    shape = np.array(np.shape(data_mat)) / k
    shape = [int(i) for i in shape]
    new_data = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_data[i, j] = data_mat[k * i, k * j]
    size = np.size(new_data)
    new_data = new_data.reshape(size)
    return new_data


def single_img2dsift(pgm_dir):
    extractor = dsift.DsiftExtractor(12, 12, 1)
    image = misc.imread(pgm_dir)
    # image = np.mean(np.double(image), axis=2) # convert RGB image into gray image
    feaArr, positions = extractor.process_image(image)
    feaArr = feaArr.reshape(np.size(feaArr))
    # IMPORTANT MODIFICATION:
    # dsift features have been APPENDED TO the original grayscale values
    return np.concatenate(((image/255).flatten(), feaArr))


def batch_convert_YaleB(truncate_num=30, images_per_person=None):
    img_suffix = 'pgm'
    mat = []
    label = []
    img_path = './CroppedYale'
    path_dir = os.listdir(img_path)
    # Create data matrix
    cats = 0
    for all_dir in path_dir:
        i = int(re.sub("\D", "", all_dir))
        file_names = os.listdir(os.path.join(img_path, all_dir))
        #mat_i = []
        imgs = 0
        for file_name in file_names:
            if file_name.find(img_suffix) != -1 and file_name.find('Ambient') == -1 and file_name.find('.bad') == -1:
                data_mat = single_img2dsift(os.path.join(img_path, all_dir, file_name))
                img_size = data_mat.shape
                mat.append(data_mat)
                label.append(i)
                imgs += 1
                if(imgs == images_per_person):
                    break
        cats += 1
        if(cats == truncate_num):
            break

        #mat.append(mat_i)
##    if images_per_person:
##        set_label = list(set(label))
##        cut_mat = []
##        cut_label = []
##        mat = np.array(mat)
##        label = np.array(label)
##        for l in set_label:
##            [cut_mat.append((mat[label == l])[i]) for i in range(images_per_person)]
##            [cut_label.append(l) for i in range(images_per_person)]
##        cut_mat = np.array(cut_mat)
##        cut_label = np.array(cut_label)
##        train_mat = cut_mat[cut_label <= truncate_num]
##        train_label = cut_label[cut_label <= truncate_num]
##        train = [train_mat, train_label]
##        test_mat = cut_mat[cut_label > truncate_num]
##        test_label = cut_label[cut_label > truncate_num]
##        test = [test_mat, test_label]
##        return train, test, img_size
    if True:
        mat = np.array(mat)
        label = np.array(label)
        train_mat = mat[label <= truncate_num]
        train_label = label[label <= truncate_num]
        train = [train_mat, train_label]
        test_mat = mat[label > truncate_num]
        test_label = label[label > truncate_num]
        test = [test_mat, test_label]
        return train, test, img_size


def main():
    mat = batch_convert_YaleB()
    print(mat)


if __name__ == "__main__":
    main()
