"""
Class for piping data from tfrecord to model. If you run this, it plots images that the model will see.

Vgg scales image in ./models/model.py.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ops import processing_ops as pops
from ops.processing_ops import Parser
import os

import param_gedi as param


class Dataspring(Parser):
    def __init__(self, p, tfrecord, verbose=False):
        super().__init__(p)
        self.tfrecord = tfrecord
        self.p = p
        self.it = None
        self.verbose = verbose

    def count_data(self):
        "Count items in tfrecord"
        print('Counting {}'.format(self.tfrecord))
        dataset_cnt = tf.data.TFRecordDataset(self.tfrecord)
        dataset_cnt = dataset_cnt.repeat(1)
        dataset_cnt = dataset_cnt.batch(1)
        cnt = dataset_cnt.reduce(0., lambda x, _: x + 1)
        return cnt

    def datagen_base(self, istraining=True, count=None):
        """

        Args:
            istraining: boolean for whether model should be trainable or not. Maybe changed later, could change some functionality in layers

        Returns:
            ds: tf dataset object

        """
        ds = tf.data.TFRecordDataset(self.tfrecord,
                                     num_parallel_reads=self.p.num_parallel_calls)  # possibly use multiple record files
        ds = ds.repeat(count)
        if istraining:
            ds = ds.shuffle(self.p.shuffle_buffer_size, reshuffle_each_iteration=True)  # shuffle up to buffer
        ds = ds.batch(self.p.BATCH_SIZE, drop_remainder=False)  # batch images, no skips
        ds = ds.map(self.tfrec_batch_parse,
                    num_parallel_calls=self.p.num_parallel_calls)  # apply parse
        # if self.p.output_size == 1:
        #     ds = ds.map(self.use_binary_lbls, self.p.num_parallel_calls)
        ds = ds.map(self.reshape_ims, num_parallel_calls=self.p.num_parallel_calls)

        # Normalization
        if self.p.histogram_eq:
            ds = ds.map(self.normalize_histeq, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.map(self.set_max_to_one_by_image, num_parallel_calls=self.p.num_parallel_calls)
        # ds = ds.map(self.rescale_im_and_clip_16bit, num_parallel_calls=self.p.num_parallel_calls)

        if self.p.augmentbool and istraining:
            ds = ds.map(self.augment, num_parallel_calls=self.p.num_parallel_calls)
            # Normalize again
            # ds = ds.map(self.cut_off_vals, num_parallel_calls=self.p.num_parallel_calls)
            ds = ds.map(self.set_max_to_one_by_image, num_parallel_calls=self.p.num_parallel_calls)

        # ds = ds.map(self.rescale_im_and_clip_renorm, num_parallel_calls=self.p.num_parallel_calls)

        if (self.p.which_model == 'vgg16') or (self.p.which_model == 'vgg19'):
            if self.verbose:
                print('Using {}'.format(self.p.which_model))
            ds = ds.map(self.make_vgg, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'mobilenet':
            if self.verbose:
                print('Using mobilenet')
            ds = ds.map(self.format_example, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'inceptionv3':
            if self.verbose:
                print('Using inceptionv3')
            ds = ds.map(self.inception_scale, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model=='resnet50':
            ds = ds.map(self.make_vgg, num_parallel_calls=self.p.num_parallel_calls)

        elif self.p.which_model == 'raw':
            if self.verbose:
                print('Using standard model')
            ds = ds.map(self.normalize_whitening, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model is not None and 'custom' in self.p.which_model:
            ds = ds.map(self.make_vgg, num_parallel_calls=self.p.num_parallel_calls)
        else:
            print('no model processing')
        ds = ds.prefetch(1)
        self.it = iter(ds)

        return ds

    @tf.function
    def datagen(self):
        """
        Generator returning parsed features
        Returns:
            imgs: batch of images
            lbls: batch of labels
            files: batch of files

        """
        imgs, lbls, files = next(self.it)
        return imgs, lbls, files

    def generator(self):
        """
        Generator returning parsed features as dictionary. Dict is used for keras model.fit
        Returns:
            X: dict
            lbls: labels

        """
        while True:
            imgs, lbls, files = next(self.it)
            X = {'input_1': imgs, 'files': files}
            yield X, lbls

    def retrain_orig_generator(self):
        """
        Generator for retraining. Original gedi model from tf 1.x isn't set up for dictionary.
        Returns:

        """
        while True:
            imgs, lbls, files = next(self.it)
            yield imgs, lbls


if __name__ == '__main__':
    p = param.Param(parent_dir='/gladstone/finkbeiner/lab/MITOPHAGY',
                    res_dir='/gladstone/finkbeiner/lab/MITOPHAGY')
    print(p.which_model)
    tfrecord = os.path.join(p.parent_dir, 'test.tfrecord')
    Dat = Dataspring(p,tfrecord)
    Dat.datagen_base(istraining=False)
    label_lst = []
    # length = Dat.count_data()
    labels = None
    length = 20
    for i in range(length):
        imgs, lbls, files = Dat.datagen()
        if labels is None:
            labels = lbls.numpy()
        else:
            labels = np.hstack((labels, lbls.numpy()))

    print(np.unique(labels, return_counts=True))
    #
    # for i in range(1):
    #     imgs, lbls, files = Dat.datagen()
    #     for img, lbl, file in zip(imgs, lbls, files):
    #
    #         lbl = lbl.numpy()
    #         img = img.numpy()
    #
    #         print(img)
    #         # print('f', file)
    #         print(np.max(img))
    #         print(np.min(img))
    #         print(img[-1,:,0]*255.0)


    if p.which_model is None:
        for i in range(1):
            imgs, lbls, files = Dat.datagen()
            for img, lbl in zip(imgs, lbls):
                plt.figure()
                lbl = lbl.numpy()
                # im = (img + 1) * 127.5
                im = np.array(img)
                im = np.reshape(im, (224,224))
                print('min',np.min(im))
                print('max', np.max(im))
                im = np.uint8(255 * im / np.max(im))
                plt.imshow(im)
                plt.title(lbl)
            plt.show()
    else:
        ###### VGG16 #######
        for i in range(3):
            imgs, lbls, files = Dat.datagen()
            for img, lbl, file in zip(imgs, lbls, files):
                lbl = lbl.numpy()
                img = img.numpy()
                img[:, :, 0] += p.VGG_MEAN[0]
                img[:, :, 1] += p.VGG_MEAN[1]
                img[:, :, 2] += p.VGG_MEAN[2]
                print(img)
                print('f', file)
                print(np.max(img))
                print(np.min(img))
                rgb = np.copy(img)
                # rgb[:, :, 0] = img[:, :, 2]
                # rgb[:, :, 2] = img[:, :, 0]
                rgb = np.float32(rgb)
                rgb *= 1
                rgb[rgb > 255] = 255
                im = np.uint8(rgb)
                plt.figure()
                plt.imshow(im)
                plt.title(lbl)
                break
            plt.show()
