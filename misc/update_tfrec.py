"""Take existing tfrec and edit it based on conditions"""
import os
import tensorflow as tf
from preprocessing.datagenerator import Dataspring
import sys
import param_gedi as param
import numpy as np
import matplotlib.pyplot as plt


class UpdateTFREC:
    def __init__(self, p, orig_tfrec):
        self.p = p
        self.Dat = Dataspring(orig_tfrec)
        self.ds = None
        self.lim = 1372  # number of samples for

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_ds(self):
        ds = tf.data.TFRecordDataset(self.Dat.tfrecord,
                                     num_parallel_reads=self.p.num_parallel_calls)  # possibly use multiple record files
        ds = ds.repeat(1)
        ds = ds.batch(1, drop_remainder=False)
        ds = ds.map(self.Dat.tfrec_batch_parse,
                    num_parallel_calls=self.p.num_parallel_calls)  # apply parse
        self.ds = ds

    def ds2record(self, tf_data_name):
        """
        Generates tfrecord in a loop.
        Args:
            tf_data_name: name of tfrecord file

        Returns:
        """
        cnt = 0
        pos_cnt = 0
        neg_cnt = 0

        with tf.io.TFRecordWriter(os.path.join(self.p.tfrec_dir, tf_data_name)) as writer:
            for img, label, filename in self.ds:
                pos = False
                neg = False
                cnt += 1
                # one less in range for matching pairs
                if not cnt % 100:
                    print('Train data:', cnt)  # Python 3 has default end = '\n' which flushes the buffer
                #                sys.stdout.flush()
                # filename = str(filepaths[i])

                # label = labels[i]
                filename = filename.numpy()[0]  # already bytes
                label = label.numpy()
                label = np.argmax(label)
                if label:
                    pos = True if pos_cnt < self.lim else False
                    neg = False
                else:
                    neg = True if neg_cnt < self.lim else False
                    pos = False

                img = img.numpy()[0, ...]
                # _img = img * 255
                # plt.imshow(_img[...,0])
                # plt.show()

                # filename = str(filename)
                # filename = str.encode(filename)
                ratio = 0.0

                if (pos and pos_cnt < self.lim) or (neg and neg_cnt < self.lim):
                    if pos:
                        pos_cnt += 1
                    if neg:
                        neg_cnt += 1
                    feature = {'label': self._int64_feature(label),
                               'ratio': self._float_feature(ratio),
                               'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                               'filename': self._bytes_feature(filename)}
                    # feature = {'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                    # 'label': self._int64_feature(label)}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                if pos_cnt >= self.lim and neg_cnt >= self.lim:
                    break

        print('Saved to ' + os.path.join(self.p.tfrec_dir, tf_data_name))

        sys.stdout.flush()


if __name__ == '__main__':
    p = param.Param()
    orig_tfrec = p.orig_test_rec
    savename = 'all_data_test_balanced.tfrecords'

    Up = UpdateTFREC(p, orig_tfrec)
    Up.get_ds()
    Up.ds2record(savename)
