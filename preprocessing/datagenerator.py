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

import param_gedi as param


class Dataspring(Parser):
    def __init__(self, tfrecord):
        super().__init__()
        self.tfrecord = tfrecord
        self.p = param.Param()
        self.it = None

    def count_data(self):
        "Count items in tfrecord"
        print('Counting {}'.format(self.tfrecord))
        dataset_cnt = tf.data.TFRecordDataset(self.tfrecord)
        dataset_cnt = dataset_cnt.repeat(1)
        dataset_cnt = dataset_cnt.batch(1)
        cnt = dataset_cnt.reduce(0., lambda x, _: x + 1)
        return cnt



    def datagen_base(self, istraining=True):
        """

        Args:
            istraining: boolean for whether model should be trainable or not. Maybe changed later, could change some functionality in layers

        Returns:
            ds: tf dataset object

        """
        ds = tf.data.TFRecordDataset(self.tfrecord,
                                     num_parallel_reads=self.p.num_parallel_calls)  # possibly use multiple record files
        ds = ds.repeat()
        if istraining:
            ds = ds.shuffle(self.p.shuffle_buffer_size, reshuffle_each_iteration=True)  # shuffle up to buffer
        ds = ds.batch(self.p.BATCH_SIZE, drop_remainder=False)  # batch images, no skips
        ds = ds.map(self.tfrec_batch_parse,
                    num_parallel_calls=self.p.num_parallel_calls)  # apply parse
        if self.p.output_size == 1:
            ds = ds.map(self.use_binary_lbls, self.p.num_parallel_calls)
        ds = ds.map(self.reshape_ims, num_parallel_calls=self.p.num_parallel_calls)

        # Normalization
        # ds = ds.map(self.set_max_to_one_by_batch, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.map(self.rescale_im_and_clip_16bit, num_parallel_calls=self.p.num_parallel_calls)

        if self.p.augmentbool and istraining:
            ds = ds.map(self.augment, num_parallel_calls=self.p.num_parallel_calls)
            # Normalize again
            ds = ds.map(self.cut_off_vals, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.map(self.rescale_im_and_clip_renorm, num_parallel_calls=self.p.num_parallel_calls)


        if (self.p.which_model == 'vgg16') or (self.p.which_model == 'vgg19'):
            print('Using {}'.format(self.p.which_model))
            ds = ds.map(self.make_vgg, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'mobilenet':
            print('Using mobilenet')
            ds = ds.map(self.format_example, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'inceptionv3':
            print('Using inceptionv3')
            ds = ds.map(self.inception_scale, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'raw':
            print('Using standard model')
            ds = ds.map(self.normalize_whitening, num_parallel_calls=self.p.num_parallel_calls)
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
    p = param.Param()
    print(p.which_model)
    tfrecord = p.data_deploy
    Chk = Dataspring(tfrecord)
    test_length = Chk.count_data().numpy()
    print(test_length)
    del Chk
    Dat = Dataspring(tfrecord)
    Dat.datagen_base(istraining=False)
    label_lst = []
    for i in range(1):
        imgs, lbls, files = Dat.datagen()
        for img, lbl, file in zip(imgs, lbls, files):

            lbl = lbl.numpy()
            img = img.numpy()

            print(img)
            # print('f', file)
            print(np.max(img))
            print(np.min(img))
            print(img[-1,:,0]*255.0)

    #
    # for i in range(1):
    #     imgs, lbls, files = Dat.datagen()
    #     for img, lbl in zip(imgs, lbls):
    #         plt.figure()
    #         lbl = lbl.numpy()
    #         # im = (img + 1) * 127.5
    #         im = np.array(img)
    #         im = np.reshape(im, (224,224,3))
    #         print(np.min(im))
    #         print(np.max(im))
    #         im = np.uint8(im)
    #         # im = np.uint8(np.reshape(im, (224, 224)))
    #         plt.imshow(im)
    #         plt.title(lbl)
    #     plt.show()
    # #

    ###### VGG16 #######
    # for i in range(3):
    #     imgs, lbls, files = Dat.datagen()
    #     for img, lbl, file in zip(imgs, lbls, files):
    #
    #         plt.figure()
    #         lbl = lbl.numpy()
    #         img = img.numpy()
    #         img[:, :, 0] += p.VGG_MEAN[0]
    #         img[:, :, 1] += p.VGG_MEAN[1]
    #         img[:, :, 2] += p.VGG_MEAN[2]
    #         print(img)
    #         print('f', file)
    #         print(np.max(img))
    #         print(np.min(img))
    #         rgb = np.copy(img)
    #         rgb[:, :, 0] = img[:, :, 2]
    #         rgb[:, :, 2] = img[:, :, 0]
    #         rgb = np.float32(rgb)
    #         rgb *= 3
    #         rgb[rgb > 255] = 255
    #         im = np.uint8(rgb)
    #         plt.imshow(im)
    #         plt.title(lbl)
    #     plt.show()
