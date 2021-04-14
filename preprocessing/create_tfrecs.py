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
import random
import os
import cv2
import matplotlib.pyplot as plt


class Record:

    def __init__(self, images_dir_live, images_dir_dead, tfrecord_dir, split, balance_method, scramble, wells):
        """

        Args:
            images_dir_live: Image direction with single label (i.e. 0)
            images_dir_dead: Image directory with different label (i.e. 1)
            tfrecord_dir: Save directory for tfrecs
            split: List to split data into training, validation, testing
            balance_method: Method to balance binary classes
            scramble: Boolean to scramble labels, the random labels on the images can help tell if the model is learning patters or memorizing samples
        """

        self.p = param.Param()
        self.images_dir_live = images_dir_live
        self.images_dir_dead = images_dir_dead
        # Add dummy folder for batch two, different tree.
        self.impaths_live = glob.glob(os.path.join(self.images_dir_live, '*.tif'))
        self.impaths_dead = glob.glob(os.path.join(self.images_dir_dead, '*.tif'))
        if len(self.impaths_dead) < len(self.impaths_live):
            self.impaths_dead, self.impaths_live = \
                self.balance_dataset(method=balance_method, smaller_lst=self.impaths_dead, larger_lst=self.impaths_live)
        self.tfrecord_dir = tfrecord_dir

        label_live = 1
        label_dead = 0

        self.labels_live = np.int16(np.ones(len(self.impaths_live)) * label_live)
        self.labels_dead = np.int16(np.ones(len(self.impaths_dead)) * label_dead)

        self._impaths = np.array(self.impaths_live + self.impaths_dead)
        self._labels = np.append(self.labels_live, self.labels_dead)
        assert len(self._impaths) == len(self._labels), 'Length of images and labels do not match.'
        assert len(self.impaths_live) + len(self.impaths_dead) == len(
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
            print('WARNING: LABELS SCRAMBLED AND INACCURATE FOR DEBUGGING')

        self.trainpaths = []
        self.valpaths = []
        self.testpaths = []
        self._split_data(wells)

    def _split_data(self, wells=None):
        if wells is None:
            length = len(self.impaths)
            self.trainpaths = self.impaths[:int(length * split[0])]
            self.valpaths = self.impaths[int(length * split[0]):int(length * (split[0] + split[1]))]
            self.testpaths = self.impaths[int(length * (split[0] + split[1])):]

            self.trainlbls = self.labels[:int(length * split[0])]
            self.vallbls = self.labels[int(length * split[0]):int(length * (split[0] + split[1]))]
            self.testlbls = self.labels[int(length * (split[0] + split[1])):]
        else:
            otherpaths = []
            otherlbls = []
            self.vallbls = []
            for imp, lbl in zip(self.impaths, self.labels):
                if os.path.basename(imp).split('_')[4] in wells:
                    self.valpaths.append(imp)
                    self.vallbls.append(lbl)
                else:
                    otherpaths.append(imp)
                    otherlbls.append(lbl)

            self.trainpaths = np.array(otherpaths[:len(otherpaths) // 2])
            self.valpaths = np.array(self.valpaths)
            self.testpaths = np.array(otherpaths[len(otherpaths) // 2:])

            self.trainlbls = np.array(otherlbls[:len(otherlbls) // 2])
            self.vallbls = np.array(self.vallbls)
            self.testlbls = np.array(otherlbls[len(otherlbls) // 2]:)


    def load_image(self, im_path):
        img = imageio.imread(im_path)
        # assume it's the correct size, otherwise resize here
        # img = cv2.resize(img, (230, 230), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def balance_dataset(self, method, smaller_lst, larger_lst):
        if len(smaller_lst) != len(larger_lst):

            if method == 'multiply':
                small_new = []
                i = 0
                while len(small_new) < len(larger_lst):
                    small_new.append(smaller_lst[i % len(smaller_lst)])
                    i += 1
                assert len(small_new) == len(larger_lst), 'lengths do not match in multiply'
                return small_new, larger_lst
            elif method == 'cutoff':
                big_new = random.sample(larger_lst, len(smaller_lst))
                assert len(smaller_lst) == len(big_new), 'lengths do not match in cutoff'
                return smaller_lst, big_new
            else:
                print('Unbalanced dataset: smaller: {}, larger: {}'.format(len(smaller_lst), len(larger_lst)))

        return smaller_lst, larger_lst


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
    # pos_dir = '/mnt/finkbeinerlab/robodata/Josh/dogs_vs_cats/train/cat'
    # neg_dir = '/mnt/finkbeinerlab/robodata/Josh/dogs_vs_cats/train/dog'
    pos_dir = '/Volumes/data/robodata/zach/pd_deep_learning/GEDI/datasets/121117iFBnBPmA/Control'
    neg_dir = '/Volumes/data/robodata/zach/pd_deep_learning/GEDI/datasets/121117iFBnBPmA/PD'
    split = [.7, .15, .15]
    balance_method = 'cutoff'
    wells = ['B11', 'B12']

    Rec = Record(pos_dir, neg_dir, p.tfrecord_dir, split, balance_method=balance_method, scramble=False, wells=wells)
    savetrain = os.path.join(p.tfrecord_dir, '121117iFBnBPmA_v2_train.tfrecord')
    saveval = os.path.join(p.tfrecord_dir, '121117iFBnBPmA_v2_val.tfrecord')
    savetest = os.path.join(p.tfrecord_dir, '121117iFBnBPmA_v2_test.tfrecord')
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
