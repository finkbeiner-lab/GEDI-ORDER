from glob import glob
import os
import numpy as np
import pandas as pd
import argparse
import imageio
import tensorflow as tf
import random
import sys
import matplotlib.pyplot as plt


class Process:
    def __init__(self, datadir, imagedir, maskdir=None):
        self.datadir = datadir
        self.imagedir = imagedir
        self.maskdir = maskdir

    def make_dataframe(self, morphology_channel):
        files = glob(os.path.join(self.imagedir, '*.tif'))
        # Make dataframe
        d = {''
             'file': [], 'filename': [], 'imrow': [], 'imcol': [], 'fov': [], 'time': [], 'plane': [], 'channel': [],
             'mask': []}
        for f in files:
            d['file'].append(f)
            filename = f.split('/')[-1]
            d['filename'].append(filename)
            d['imrow'].append(self.get_row(filename))
            d['imcol'].append(self.get_col(filename))
            d['fov'].append(self.get_fov(filename))
            d['time'].append(self.get_time(filename))
            d['plane'].append(self.get_plane(filename))
            channel = self.get_channel(filename)
            d['channel'].append(channel)
            if self.maskdir is not None and channel == morphology_channel:
                stem = filename.split('.')[0]
                mfiles = glob(os.path.join(self.maskdir, stem + '*.tif'))
                assert len(mfiles) <= 1
                if len(mfiles) == 0:
                    d['mask'].append('null')
                else:
                    d['mask'].append(mfiles[0])
            else:
                d['mask'].append('null')
        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.datadir, 'data.csv'))
        return df

    def split_train_val_test(self, df, split=[.8, .1, .1]):
        """
        Split to train val test by groups
        :param df:
        :return:
        """
        img_groups = df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        a = np.arange(img_groups.ngroups)
        np.random.seed(121)
        np.random.shuffle(a)
        train_df = df[img_groups.ngroup().isin(a[:int(len(img_groups) * split[0])])]
        df = df.drop(train_df.index)

        img_groups = df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        a = np.arange(img_groups.ngroups)
        np.random.seed(121)
        np.random.shuffle(a)
        val_df = df[img_groups.ngroup().isin(a[:int(len(img_groups) * split[1] / (1 - split[0]))])]
        test_df = df.drop(val_df.index)
        return train_df, val_df, test_df

    def select_df(self, df, fovs=[], planes=[], channels=[], rows=None, cols=None):
        if rows is not None and cols is not None:
            df = df[(df.fov.isin(fovs)) & (df.plane.isin(planes)) & (df.channel.isin(channels)) &
                    (df.imrow.isin(rows)) & (df.imcol.isin(cols))]
        else:
            df = df[(df.fov.isin(fovs)) & (df.plane.isin(planes)) & (df.channel.isin(channels))]
        return df

    def get_row(self, filename):
        row = int(filename[:3])
        return row

    def get_col(self, filename):
        col = int(filename[3:6])
        return col

    def get_fov(self, filename):
        return int(filename[7])

    def get_time(self, filename):
        return int(filename[9:12])

    def get_plane(self, filename):
        return int(filename[12:15])

    def get_channel(self, filename):
        return int(filename[15:18])


class DataframeToCrops:

    def __init__(self, df, cropdir, label_dict):

        self.cropdir = cropdir
        self.img_groups = df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        self.idx = list(self.img_groups)
        _label_dict = {(7, 1): 1, (8, 1): 1,
                       (7, 12): 2, (8, 12): 2,
                       (1, 1): 3, (2, 1): 3,
                       (1, 12): 4, (2, 12): 4,
                       (3, 1): 5, (4, 1): 5,
                       (3, 12): 6, (4, 12): 6,
                       (1, 10): 7, (2, 10): 7
                       }
        self.label_dict = label_dict if label_dict is not None else _label_dict

    def get_and_save_crops(self):
        def tuple_to_string(tup):
            res = ''
            for i in tup:
                res += str(i)
            return res

        for key, group in self.idx:
            # group = self.img_groups.get_group(key)
            key_string = tuple_to_string(key)
            class_label = self.label_dict[(key[0], key[1])]
            if not os.path.exists(os.path.join(self.cropdir, str(class_label))): os.makedirs(
                os.path.join(self.cropdir, str(class_label)))
            group = group.sort_values(by='channel')
            img_paths = group.file.tolist()
            imgs = []
            for img_path in img_paths:
                _im = imageio.v3.imread(img_path)
                _im = (_im / np.max(_im) * 255).astype('uint8')
                imgs.append(_im)
            img = np.dstack(imgs)
            # get crops
            cropsize = 300
            for i in range(0, np.shape(img)[0]-cropsize, cropsize):
                for j in range(0, np.shape(img)[1]-cropsize, cropsize):
                    crop = img[i:i + cropsize, j:j + cropsize]
                    # plt.imshow(crop)
                    # plt.show()
                    croppath = os.path.join(self.cropdir, str(class_label), f'mito_{key_string}_{i}_{j}.tif')
                    imageio.imwrite(croppath, crop)


class CropsToRecord:
    def __init__(self, datadir, split, tfrecord_dir):
        self.tfrecord_dir = tfrecord_dir
        if not os.path.exists(self.tfrecord_dir): os.makedirs(self.tfrecord_dir)

        self.class_labels = [os.path.basename(f) for f in glob(os.path.join(datadir, '*'))]
        self.labels = []
        self.impaths = []
        for lbl in self.class_labels:
            fpaths = glob(os.path.join(datadir, lbl, '*'))
            self.labels += [lbl] * len(fpaths)
            self.impaths += fpaths
        if not len(self.impaths) == len(self.labels): raise Exception('labels and impaths must be same length')
        self.impaths = np.array(self.impaths)
        self.labels = np.array(self.labels)
        self.shuffled_idx = np.arange(len(self.impaths))
        self.scrambled_idx = self.shuffled_idx.copy()
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        self.impaths = self.impaths[self.shuffled_idx]
        self.labels = self.labels[self.shuffled_idx]

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
                    print('Processed data:', i)  # Python 3 has default end = '\n' which flushes the buffer
                #                sys.stdout.flush()
                filename = str(filepaths[i])

                img = self.load_image(filename)

                label = int(labels[i])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/')
    parser.add_argument('--imagedir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/Images for MitophAIgy Test')
    parser.add_argument('--maskdir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/Masks')
    parser.add_argument('--cropdir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/Crops')
    parser.add_argument('--tfrecord_dir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY')
    args = parser.parse_args()
    print('ARGS: ', args)
    Proc = Process(args.datadir, args.imagedir, args.maskdir)
    df = Proc.make_dataframe(morphology_channel=3)
    # df = Proc.select_df(df, fovs=[1], planes=[1], channels=[1], rows=None, cols=None)
    # print(df)

    DC = DataframeToCrops(df, args.cropdir, label_dict=None)
    DC.get_and_save_crops()

    Rec = CropsToRecord(datadir=args.cropdir, split=[0.7, 0.15, 0.15], tfrecord_dir=args.tfrecord_dir)
    savetrain = os.path.join(args.tfrecord_dir, 'train.tfrecord')
    saveval = os.path.join(args.tfrecord_dir, 'val.tfrecord')
    savetest = os.path.join(args.tfrecord_dir, 'test.tfrecord')
    Rec.tiff2record(savetrain, Rec.trainpaths, Rec.trainlbls)
    Rec.tiff2record(saveval, Rec.valpaths, Rec.vallbls)
    Rec.tiff2record(savetest, Rec.testpaths, Rec.testlbls)
