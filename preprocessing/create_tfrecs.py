#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:19:32 2019

@author: joshlamstein
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert images to TF Records.

Each record for matching has:
    Image
    Mask
    Filepath of image (for tracking it down if there's a problem)
    label - id number of neuron
    timepoint - time point of neuron

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

To do:
    Add img2 and mask2 to pair the data for the siamese network. 
    Save into tf record. 
    Simplest is to save by timepoint, 1->2, 2->3, end -> rand and flag skip.
    Actually simplest to train by family ID. But that isn't the target. 
    Make another tfrecord that programs in negative random samples, verifying that they're not the same family. 

"""

import tensorflow as tf
import imageio
import numpy as np
import sys
import glob
import param_gedi as param
import os


class Record:

    def __init__(self, images_dir_A, images_dir_B, tfrecord_dir, split, scramble):
        """

        Args:
            images_dir_A: Image direction with single label (i.e. 0)
            images_dir_B: Image directory with different label (i.e. 1)
            tfrecord_dir: Save directory for tfrecs
            split: List to split data into training, validation, testing
            scramble: Boolean to scramble labels, the random labels on the images can help tell if the model is learning patters or memorizing samples
        """

        self.p = param.Param()
        self.images_dir_A = images_dir_A
        self.images_dir_B = images_dir_B
        # Add dummy folder for batch two, different tree.
        self.impaths_A = glob.glob(os.path.join(self.images_dir_A, '*', '*', '*.tif'))
        self.impaths_B = glob.glob(os.path.join(self.images_dir_B, '*', '*', '*.tif'))

        self.tfrecord_dir = tfrecord_dir
        # ../images_dir_A/positive
        # ../images_dir_A/negative
        positive_negative_A = images_dir_A.split('/')[-2]
        if positive_negative_A == 'positive':
            label_A = 1
        elif positive_negative_A == 'negative':
            label_A = 0
        else:
            raise ValueError('Last folder A in image directory must be either \'positive\' or \'negative\'.')

        positive_negative_B = images_dir_B.split('/')[-2]
        if positive_negative_B == 'positive':
            label_B = 1
        elif positive_negative_B == 'negative':
            label_B = 0
        else:
            raise ValueError('Last folder B in image directory must be either \'positive\' or \'negative\'.')

        self.labels_A = np.int16(np.ones(len(self.impaths_A)) * label_A)
        self.labels_B = np.int16(np.ones(len(self.impaths_B)) * label_B)

        self._impaths = np.array(self.impaths_A + self.impaths_B)
        self._labels = np.append(self.labels_A, self.labels_B)
        assert len(self._impaths) == len(self._labels), 'Length of images and labels do not match.'
        assert len(self.impaths_A) + len(self.impaths_B) == len(
            self._impaths), 'Summed lengths of image paths do not match'
        self.shuffled_idx = np.arange(len(self._impaths))
        self.scrambled_idx = self.shuffled_idx.copy()
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        print(self.shuffled_idx)

        self.impaths = self._impaths[self.shuffled_idx]
        if not scramble:
            self.labels = self._labels[self.shuffled_idx]
        else:
            np.random.seed(1)

            np.random.shuffle(self.scrambled_idx)
            self.labels = self._labels[self.scrambled_idx]

        length = len(self.impaths)

        self.trainpaths = self.impaths[:int(length * split[0])]
        self.valpaths = self.impaths[int(length * split[0]):int(length * (split[0] + split[1]))]
        self.testpaths = self.impaths[int(length * (split[0] + split[1])):]

        self.trainlbls = self.labels[:int(length * split[0])]
        self.vallbls = self.labels[int(length * split[0]):int(length * (split[0] + split[1]))]
        self.testlbls = self.labels[int(length * (split[0] + split[1])):]

    def load_image(self, im_path):
        img = imageio.imread(im_path)
        # assume it's the correct size, otherwise resize here
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def tiff2record(self, tf_data_name, filepaths, labels):
        """
        Generates tfrecord in a loop.
        Args:
            tf_data_name: name of tfrecord file

        Returns:
        """
        assert len(filepaths) == len(labels), 'len of filepaths and labels do not match {} {}'.format(len(filepaths),
                                                                                                      len(labels))
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
                ratio = 0.0

                feature = {'label': self._int64_feature(label),
                           'ratio': self._float_feature(ratio),
                           'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'filename': self._bytes_feature(filename)}
                # feature = {'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                # 'label': self._int64_feature(label)}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print('Saved to ' + os.path.join(self.tfrecord_dir, tf_data_name))

        sys.stdout.flush()


if __name__ == '__main__':
    p = param.Param()
    pos_dir = '/mnt/data/MJFOX/batch2_3_3_2020-02-26-18-21-17/positive/Images'
    neg_dir = '/mnt/data/MJFOX/batch2_3_3_2020-02-26-18-21-17/negative/Images'
    neg_dir = '/mnt/data/MJFOX/Crops/negative/Images'
    split = [.7, .15, .15]

    Rec = Record(pos_dir, neg_dir, p.tfrecord_dir, split, scramble=False)
    savetrain = os.path.join(p.tfrecord_dir, 'batch2_train_3_3_100_scrambled_labels.tfrecord')
    saveval = os.path.join(p.tfrecord_dir, 'batch2_val_3_3_100_scrambled_labels.tfrecord')
    savetest = os.path.join(p.tfrecord_dir, 'batch2_test_3_3_100_scrambled_labels.tfrecord')
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
