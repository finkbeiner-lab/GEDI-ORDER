"""
Process tfrecord

Run to visualize tfrecords. If normalizing, unnormalize.

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
        dataset_cnt = tf.data.TFRecordDataset(self.tfrecord)
        dataset_cnt = dataset_cnt.repeat(1)
        cnt = dataset_cnt.reduce(0., lambda x, _: x + 1)
        return cnt

    def reshape_ims(self, imgs, lbls, files):
        if self.p.orig_size[-1] > self.p.target_size[-1]:
            # Remove alpha channel
            channels = tf.unstack(imgs, axis=-1)
            imgs = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
        if self.p.orig_size[0] > self.p.target_size[0]:
            y0 = (self.p.orig_size[0] - self.p.target_size[0]) // 2
            x0 = (self.p.orig_size[1] - self.p.target_size[1]) // 2

            imgs = tf.image.crop_to_bounding_box(imgs, y0, x0, self.p.target_size[1], self.p.target_size[0])
        return imgs, lbls, files

    def datagen_base(self, istraining=True):
        """
        Generator for extracting data from tfrecords

        Args:
            path: Path(s) to tfrecords
            batch_size: ..
            buffer_size: Shuffle buffer size; no shuffle if 1
            row_parser: Method to parse single record
            img_parser: Method to transform images individually
            transform_params: Args to ImageDataGenerator augmentation constructor
            parallel_reads: ..

        Returns:
            Is a generator; each next call gives numpy tensor of shape (batch_size, img_height, img_width, channels)

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

        # Divides by 255
        ds = ds.map(self.set_max_to_one, num_parallel_calls=self.p.num_parallel_calls)

        if self.p.augmentbool and istraining:
            ds = ds.map(self.augment, num_parallel_calls=self.p.num_parallel_calls)
            ds = ds.map(self.cut_off_vals, num_parallel_calls=self.p.num_parallel_calls)

        if (self.p.which_model == 'vgg16') or (self.p.which_model == 'vgg19'):
            print('Using {}'.format(self.p.which_model))
            ds = ds.map(self.make_vgg, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'mobilenet':
            print('Using mobilenet')
            ds = ds.map(self.format_example, num_parallel_calls=self.p.num_parallel_calls)
        elif self.p.which_model == 'inceptionv3':
            print('Using inceptionv3')
            ds = ds.map(self.inception_scale, num_parallel_calls=self.p.num_parallel_calls)
        else:
            print('Using standard model')
            ds = ds.map(self.normalize_whitening, num_parallel_calls=self.p.num_parallel_calls)
        ds = ds.prefetch(1)
        self.it = iter(ds)

        return ds

    @tf.function
    def datagen(self):
        # lbls, imgs, filenames = next(self.it)
        imgs, lbls, files = next(self.it)
        return imgs, lbls, files

    def generator(self):
        while True:
            imgs, lbls, files = next(self.it)
            X = {'imgs': imgs, 'files': files}
            yield X, lbls


if __name__ == '__main__':
    p = param.Param()
    print(p.which_model)
    tfrecord = p.data_val
    Chk = Dataspring(tfrecord)
    test_length = Chk.count_data().numpy()
    print(test_length)
    del Chk
    Dat = Dataspring(tfrecord)
    Dat.datagen_base(istraining=True)
    label_lst = []
    # for i in range(int(test_length//p.BATCH_SIZE)):
    #     imgs, lbls = Dat.datagen()
    #     labels = list(np.argmax(lbls, axis=1))
    #     label_lst.append(labels)
    #
    # print(np.mean(label_lst))

    # for i in range(1):
    #     imgs, lbls, files = Dat.datagen()
    #     for img, lbl in zip(imgs, lbls):
            # plt.figure()
        #     lbl = lbl.numpy()
        #     # im = (img + 1) * 127.5
        #     im = np.array(img)
        #     print(np.min(im))
        #     print(np.max(im))
        #     im *= 255
        #     im = np.uint8(im)
        #     # im = np.uint8(np.reshape(im, (224, 224)))
        #     plt.imshow(im)
        #     plt.title(lbl)
        # plt.show()

    for i in range(1):
        imgs, lbls, files = Dat.datagen()
        for img, lbl in zip(imgs, lbls):
            plt.figure()
            lbl = lbl.numpy()
            img = img.numpy()
            img[:,:,0] += p.VGG_MEAN[0]
            img[:,:,1] += p.VGG_MEAN[1]
            img[:,:,2] += p.VGG_MEAN[2]
            print(np.max(img))
            print(np.min(img))
            rgb = np.copy(img)
            rgb[:,:,0] = img[:,:,2]
            rgb[:,:,2] = img[:,:,0]
            im = np.uint8(rgb)
            plt.imshow(im)
            plt.title(lbl)
        plt.show()