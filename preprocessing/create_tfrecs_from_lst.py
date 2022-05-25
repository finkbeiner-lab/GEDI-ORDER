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
from utils.utils import get_timepoint
import random


class Record:

    def __init__(self, images_lst_live, images_lst_dead, tfrecord_dir, split, balance_method):
        """
        Class for building tfrecords. Takes lists, combines them using cutoff or multiply balance methods.
        Args:
            images_lst_live: Image lst with liv labels (1)
            images_lst_dead: Image list with dead labels (0)
            tfrecord_dir: Save directory for tfrecs
            split: List to split data into training, validation, testing
            balance_method: Method to balance binary classes, 'cutoff' - remove data, 'multiply' - duplicate smaller data class to match larger class, None - leave unbalanced
        """
        assert isinstance(images_lst_dead, list), 'images_lst_dead must be list'
        self.p = param.Param()

        self.tfrecord_dir = tfrecord_dir
        self.impaths_live = images_lst_live
        self.impaths_dead = images_lst_dead
        self.balance_method = balance_method
        assert balance_method=='cutoff' or balance_method=='multiply' or balance_method is None, 'invalid entry for balance method'
        if self.balance_method == 'multiply':
            trainlive, vallive, testlive, traindead, valdead, testdead = self.multiply_dataset(split)
        elif self.balance_method == 'cutoff':
            self.impaths_live, self.impaths_dead = \
                self.balance_dataset(method=self.balance_method, lista=self.impaths_live, listb=self.impaths_dead)
            livelen = len(self.impaths_live)
            deadlen = len(self.impaths_dead)
            trainlive = self.impaths_live[:int(livelen * split[0])]
            vallive = self.impaths_live[int(livelen * split[0]):int(livelen * (split[0] + split[1]))]
            testlive = self.impaths_live[int(livelen * (split[0] + split[1])):]
            traindead = self.impaths_dead[:int(deadlen * split[0])]
            valdead = self.impaths_dead[int(deadlen * split[0]):int(deadlen * (split[0] + split[1]))]
            testdead = self.impaths_dead[int(deadlen * (split[0] + split[1])):]

        self.trainpaths, self.trainlbls = self.label_and_shuffle(trainlive, traindead)
        self.valpaths, self.vallbls = self.label_and_shuffle(vallive, valdead)
        self.testpaths, self.testlbls = self.label_and_shuffle(testlive, testdead)
        # self.labels_live = np.int16(np.ones(len(self.impaths_live)) * label_live)
        # self.labels_dead = np.int16(np.ones(len(self.impaths_dead)) * label_dead)
        #
        # self._impaths = np.array(self.impaths_live + self.impaths_dead)
        # self._labels = np.append(self.labels_live, self.labels_dead)
        # assert len(self._impaths) == len(self._labels), 'Length of images and labels do not match.'
        # assert len(self.impaths_live) + len(self.impaths_dead) == len(
        #     self._impaths), 'Summed lengths of image paths do not match'
        # self.shuffled_idx = np.arange(len(self._impaths))
        # self.scrambled_idx = self.shuffled_idx.copy()
        # np.random.seed(0)
        # np.random.shuffle(self.shuffled_idx)
        # # print(self.shuffled_idx)
        #
        # self.impaths = self._impaths[self.shuffled_idx]
        # self.labels = self._labels[self.shuffled_idx]
        #
        # length = len(self.impaths)
        #
        # self.trainpaths = self.impaths[:int(length * split[0])]
        # self.valpaths = self.impaths[int(length * split[0]):int(length * (split[0] + split[1]))]
        # self.testpaths = self.impaths[int(length * (split[0] + split[1])):]
        #
        # self.trainlbls = self.labels[:int(length * split[0])]
        # self.vallbls = self.labels[int(length * split[0]):int(length * (split[0] + split[1]))]
        # self.testlbls = self.labels[int(length * (split[0] + split[1])):]
        #

    def label_and_shuffle(self, livelst, deadlst):
        """
        Assign 1 for positive and 0 for negative class. Shuffle.
        Args:
            livelst:
            deadlst:

        Returns:

        """
        livelbls = [1 for _ in livelst]
        deadlbls = [0 for _ in deadlst]
        impaths = np.array(livelst + deadlst)
        lbls = np.int16(np.array(livelbls + deadlbls))
        shuffled_idx = np.arange(len(impaths))
        chk_shuffle = shuffled_idx.copy()
        np.random.seed(0)
        np.random.shuffle(shuffled_idx)
        assert shuffled_idx[0] != chk_shuffle[0], 'check shuffling'
        assert len(lbls) == len(impaths), 'label and image path lengths do not match'
        impaths = impaths[shuffled_idx]
        lbls = lbls[shuffled_idx]
        return impaths, lbls

    def load_image(self, im_path):
        img = imageio.imread(im_path)
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def multiply_dataset(self, split):
        """
        Multiply images
        Args:
            split: array of floats for percentage to split dataset

        Returns:

        """
        livelen = len(self.impaths_live)
        deadlen = len(self.impaths_dead)
        trainlive = self.impaths_live[:int(livelen * split[0])]
        vallive = self.impaths_live[int(livelen * split[0]):int(livelen * (split[0] + split[1]))]
        testlive = self.impaths_live[int(livelen * (split[0] + split[1])):]
        traindead = self.impaths_dead[:int(deadlen * split[0])]
        valdead = self.impaths_dead[int(deadlen * split[0]):int(deadlen * (split[0] + split[1]))]
        testdead = self.impaths_dead[int(deadlen * (split[0] + split[1])):]

        trainlive, traindead = self.balance_dataset(self.balance_method, trainlive, traindead)
        vallive, valdead = self.balance_dataset(self.balance_method, vallive, valdead)
        testlive, testdead = self.balance_dataset(self.balance_method, testlive, testdead)
        return trainlive, vallive, testlive, traindead, valdead, testdead

    def balance_dataset(self, method, lista, listb):
        """
        Balance dataset using cutoff or multiply methods.
        Args:
            method:
            lista:
            listb:

        Returns:

        """
        if len(lista) < len(listb):

            if method == 'multiply':
                small_new = []
                i = 0
                while len(small_new) < len(listb):
                    small_new.append(lista[i % len(lista)])
                    i += 1
                assert len(small_new) == len(listb), 'lengths do not match in multiply'
                return small_new, listb
            elif method == 'cutoff':
                big_new = random.sample(listb, len(lista))
                assert len(lista) == len(big_new), 'lengths do not match in cutoff'
                return lista, big_new
            else:
                print('Unbalanced dataset: smaller: {}, larger: {}'.format(len(lista), len(listb)))
        elif len(lista) > len(listb):
            if method == 'multiply':
                small_new = []
                i = 0
                while len(small_new) < len(lista):
                    small_new.append(listb[i % len(listb)])
                    i += 1
                assert len(small_new) == len(lista), 'lengths do not match in multiply'
                return lista, small_new
            elif method == 'cutoff':
                big_new = random.sample(lista, len(listb))
                assert len(listb) == len(big_new), 'lengths do not match in cutoff'
                return big_new, listb
            else:
                print('Unbalanced dataset: smaller: {}, larger: {}'.format(len(lista), len(listb)))
        return lista, listb

    def tiff2record(self, tf_data_name, filepaths, labels):
        """
        Generates tfrecord.
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
                    print('Processed data:', i)  # Python 3 has default end = '\n' which flushes the buffer
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
    pos_dir = '/mnt/finkbeinerlab/robodata/JeremyTEMP/GalaxyTEMP/LINCS072017RGEDI-A/LiveVoronoi'
    neg_dir = '/mnt/finkbeinerlab/robodata/JeremyTEMP/GalaxyTEMP/LINCS072017RGEDI-A/DeadVoronoi'
    split = [.7, .15, .15]
    _poss = glob.glob(os.path.join(pos_dir, '*.tif'))
    _negs = glob.glob(os.path.join(neg_dir, '*.tif'))
    poss = [f for f in _poss if get_timepoint(f) < 11]
    negs = [f for f in _negs if get_timepoint(f) < 11]

    if len(poss) > len(negs):
        poss = random.sample(poss, len(negs))
        assert len(poss) == len(negs), 'expect negative and positive list to be same length'

    Rec = Record(poss, negs, p.tfrecord_dir, split, balance_method='cutoff')
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # b = [1, 2, 3, 4, 5, 6]
    # aa, bb = Rec.balance_dataset('cutoff', a, b)
    # print('aa',aa)
    # print('bb',bb)
    # bb, aa = Rec.balance_dataset('cutoff', b, a)
    # print('2')
    # print('aa', aa)
    # print('bb', bb)
    savetrain = os.path.join(p.tfrecord_dir, 'LINCS072017RGEDI-A_train.tfrecord')
    saveval = os.path.join(p.tfrecord_dir, 'LINCS072017RGEDI-A_val.tfrecord')
    savetest = os.path.join(p.tfrecord_dir, 'LINCS072017RGEDI-A_test.tfrecord')
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
