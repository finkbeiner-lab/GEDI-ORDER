"""
Explore data in tf record
todo: set datagenerator.py to output raw images for visualization for measuring. Record gedi signal
"""

import param_gedi as param
from preprocessing.datagenerator import Dataspring
import numpy as np
import cv2


def get_mean_of_img_batch(images):
    images = np.mean(images, axis=3)
    images = np.mean(images, axis=2)
    images = np.mean(images, axis=1)
    return images

def get_batch_hist(imgs):
    """

    Returns:

    """
    np.histogram()


p = param.Param()
tfrecord = p.data_deploy
tfrecord = '/mnt/finkbeinernas/robodata/Josh/GEDI-ORDER/testH23.tfrecord'
Dat = Dataspring(p, tfrecord)
Dat.datagen_base(istraining=False)
Chk = Dataspring(p, tfrecord)
test_length = Chk.count_data().numpy()
print(test_length)
del Chk
for i in range(test_length // p.BATCH_SIZE):
    imgs, lbls, files = Dat.datagen()
    imgs = get_mean_of_img_batch(imgs)

def get_gedi_signal():
    pass