from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors as colors
import matplotlib.cm as cmx
from array2gif import write_gif


data = loadmat("../!Important/figures/AE_big")
images_raw_duck = data['H0']
images_HM2_duck = data['HM2']
reconstr_duck = data['HM']
pca_duck = data['PCA']
pca_reconstr_duck = data['PCAr']

def scale(images, scale):
    scaled = np.zeros((images.shape[0], scale*images.shape[1], scale*images.shape[2]), dtype=images.dtype)
    for i in range(images.shape[0]):
        scaled[i] = cv2.resize(images[i], (scale*images.shape[2], scale*images.shape[1]), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    return scaled

inferno = plt.get_cmap('inferno')

def transform(images, x, y, flip=False, scale=1):
    if flip:
        string = 'ilkj->ijkl'
        shape = (-1, x, y, 4)
    else:
        string = 'iklj->ijkl'
        shape = (-1, y, x, 4)
    cNorm  = colors.Normalize(vmin=np.min(images)/scale, vmax=np.max(images)/scale)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=inferno)
    return np.einsum(string, 256*scalarMap.to_rgba(images.reshape(-1, x*y)).reshape(shape)[:, :, :, :3]).astype(np.uint8)

s = 10
images_HM2_duck_inflated = scale(images_HM2_duck.reshape(-1, 15, 20), s).reshape(-1, 20*15*s*s)

write_gif(transform(images_raw_duck, 128, 128), '../!Important/figures/H0.gif')
write_gif(transform(images_HM2_duck_inflated, 20*s, 15*s, scale=1), '../!Important/figures/HM2.gif')
write_gif(transform(reconstr_duck, 128, 128), '../!Important/figures/HM.gif')