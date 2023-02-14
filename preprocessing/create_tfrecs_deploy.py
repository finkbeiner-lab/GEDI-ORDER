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

    def __init__(self, images_dir_A, tfrecord_dir, lbl):
        self.images_dir_A = images_dir_A
        # Add dummy folder for batch two, different tree.
        imgs = glob.glob(os.path.join(self.images_dir_A, '*.tif'))
        if len(imgs)==0:
            imgs = glob.glob(os.path.join(self.images_dir_A, '**', '*.tif'))
        self.impaths_A = imgs
        self.tfrecord_dir = tfrecord_dir

        self.impaths = np.array(self.impaths_A)
        self.lbls = np.array([lbl for i in range(len(self.impaths_A))])
        assert len(self.impaths) == len(self.lbls), 'Length of images and labels do not match.'

    def load_image(self, im_path):
        img = imageio.imread(im_path)
        # assume it's the correct size, otherwise resize here
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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
                    print('Deploy data:', i)  # Python 3 has default end = '\n' which flushes the buffer
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
                #            'label': self._int64_feature(label)}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print('Saved to ' + os.path.join(self.tfrecord_dir, tf_data_name))

        sys.stdout.flush()


if __name__ == '__main__':
    p = param.Param()
    lbl = 1
    if lbl ==1:
        lblstr = 'positive'
    elif lbl==0:
        lblstr= 'negative'
    use_dir = '/mnt/finkbeinerlab/robodata/GalaxyTEMP/BSMachineLearning_TestCuration/batches/5'

    Rec = Record(use_dir,  p.tfrecord_dir, lbl)
    savedeploy = os.path.join(p.tfrecord_dir, use_dir.split('/')[-3] + '_' +use_dir.split('/')[-1] + '.tfrecord')
    Rec.tiff2record(savedeploy, Rec.impaths, Rec.lbls)
