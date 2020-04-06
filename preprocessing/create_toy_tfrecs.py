"""
Toy data tf records cats vs dogs
"""

import tensorflow as tf
import imageio
import numpy as np
import sys
import glob
import param_gedi as param
import os
import cv2
import matplotlib.pyplot as plt


class Record:

    def __init__(self, images_dir_A, images_dir_B, tfrecord_dir, split):
        cats = []
        dogs = []
        self.tfrecord_dir = tfrecord_dir

        _files = glob.glob(os.path.join('/mnt/finkbeinerlab/robodata/Josh/dogs_vs_cats/train/*.jpg'))
        for f in _files:
            parts = f.split('/')[-1].split('.')
            if parts[0] == 'cat':
                cats.append(f)
            elif parts[0] == 'dog':
                dogs.append(f)
        np.random.shuffle(cats)
        np.random.shuffle(dogs)
        self.impaths_A = list(cats)
        self.impaths_B = list(dogs)
        label_A = 1
        label_B = 0

        self.labels_A = np.int16(np.ones(len(self.impaths_A)) * label_A)
        self.labels_B = np.int16(np.ones(len(self.impaths_B)) * label_B)

        self._impaths = np.array(self.impaths_A + self.impaths_B)
        self._labels = np.append(self.labels_A, self.labels_B)
        assert len(self._impaths) == len(self._labels), 'Length of images and labels do not match.'
        assert len(self.impaths_A) + len(self.impaths_B) == len(
            self._impaths), 'Summed lengths of image paths do not match'
        self.shuffled_idx = np.arange(len(self._impaths))
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        print(self.shuffled_idx)

        self.impaths = self._impaths[self.shuffled_idx]
        self.labels = self._labels[self.shuffled_idx]
        assert self.impaths[0] != self._impaths[0], 'check randomization'

        length = len(self.impaths)

        self.trainpaths = self.impaths[:int(length*split[0])]
        self.valpaths = self.impaths[int(length*split[0]):int(length * (split[0] + split[1]))]
        self.testpaths = self.impaths[int(length * (split[0] + split[1])):]

        self.trainlbls = self.labels[:int(length*split[0])]
        self.vallbls = self.labels[int(length*split[0]):int(length * (split[0] + split[1]))]
        self.testlbls = self.labels[int(length * (split[0] + split[1])):]

    def load_image(self, im_path):
        img = cv2.imread(im_path)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(img)
        # plt.show()
        # assume it's the correct size, otherwise resize here
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def tiff2record(self, tf_data_name, filepaths, labels):
        """
        Generates tfrecord in a loop.
        Args:
            tf_data_name: name of tfrecord file

        Returns:
        """
        assert len(filepaths)==len(labels), 'len of filepaths and labels do not match {} {}'.format(len(filepaths), len(labels))
        with tf.io.TFRecordWriter(os.path.join(self.tfrecord_dir, tf_data_name)) as writer:
            for i in range(len(filepaths)):
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i)  # Python 3 has default end = '\n' which flushes the buffer
                #                sys.stdout.flush()
                filename = str(filepaths[i])

                img = self.load_image(filename)

                label = labels[i]
                filename = str(filename)
                filename = str.encode(filename)

                # feature = {'label': self._int64_feature(label),
                # 'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                # 'filename': self._bytes_feature(filename)}
                feature = {'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'label': self._int64_feature(label)}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print('Saved to ' + os.path.join(self.tfrecord_dir, tf_data_name))

        sys.stdout.flush()

if __name__ == '__main__':
    p = param.Param()
    pos_dir = '/mnt/data/MJFOX/Crops/positive/DNE'
    neg_dir = '/mnt/data/MJFOX/Crops/negative/DNE'
    split = [.7, .15, .15]

    Rec = Record(pos_dir, neg_dir, p.tfrecord_dir, split)
    savetrain = os.path.join(p.tfrecord_dir, 'train_catsdogs.tfrecord')
    saveval = os.path.join(p.tfrecord_dir, 'val_catsdogs.tfrecord')
    savetest = os.path.join(p.tfrecord_dir, 'test_catsdogs.tfrecord')
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)