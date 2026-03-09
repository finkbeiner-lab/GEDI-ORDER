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

    def __init__(self, images_lst_live, images_lst_dead, tfrecord_dir, split, balance_method,
                 split_method='percentage', tile_channel='Confocal',
                 train_tiles=None, val_tiles=None, test_tiles=None):
        """
        Class for building tfrecords. Takes lists, combines them using cutoff or multiply balance methods.
        Args:
            images_lst_live: Image lst with live labels (1)
            images_lst_dead: Image list with dead labels (0)
            tfrecord_dir:    Save directory for tfrecs
            split:           List to split data into training, validation, testing
            balance_method:  'cutoff' | 'multiply' | None
            split_method:    'percentage' for random split, 'tiles' for tile-ID-based split
            tile_channel:    Channel name in tile pattern e.g. 'Confocal' or 'Epi' (used when split_method='tiles')
            train_tiles:     List of tile numbers for training (used when split_method='tiles')
            val_tiles:       List of tile numbers for validation (used when split_method='tiles')
            test_tiles:      List of tile numbers for test (used when split_method='tiles')
        """
        assert isinstance(images_lst_dead, list), 'images_lst_dead must be list'
        self.p = param.Param()

        self.tfrecord_dir = tfrecord_dir
        self.impaths_live = images_lst_live
        self.impaths_dead = images_lst_dead
        self.balance_method = balance_method

        if split_method == 'tiles':
            # Use tile-ID-based splitting
            trainlive, vallive, testlive, traindead, valdead, testdead = self.split_by_tile_id(
                self.impaths_live, self.impaths_dead,
                channel=tile_channel,
                train_nums=train_tiles,
                val_nums=val_tiles,
                test_nums=test_tiles)
        else:
            # Use original percentage-based splitting
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

        Returns:a

        """
        print(f"DEBUG: First 3 positive images: {livelst[:3]}")
        print(f"DEBUG: First 3 negative images: {deadlst[:3]}")
        print(f"DEBUG: len(livelst) = {len(livelst)}, len(deadlst) = {len(deadlst)}")

        livelbls = [1 for _ in livelst]
        deadlbls = [0 for _ in deadlst]
        impaths = np.array(livelst + deadlst)
        lbls = np.int16(np.array(livelbls + deadlbls))

        print(f"DEBUG: Total images = {len(impaths)}")
        if len(impaths) == 0:
            raise ValueError("No image paths were loaded. Check pos_dir and neg_dir contents.")
        
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
        try:
            # Check if file exists and is not empty
            import os
            if not os.path.exists(im_path) or os.path.getsize(im_path) == 0:
                print(f"Warning: Skipping empty or non-existent file: {im_path}")
                return None

            img = imageio.imread(im_path)
            img = img.astype(np.float32)

            # Ensure consistent dimensions - resize to orig_size if different
            expected_shape = self.p.orig_size
            if len(img.shape) == 2:  # Grayscale image
                img = img.reshape(img.shape + (1,))

            if img.shape != expected_shape:
                import cv2
                # Resize to expected dimensions
                img = cv2.resize(img.squeeze(), (expected_shape[1], expected_shape[0]))
                if len(expected_shape) == 3 and expected_shape[2] == 1:
                    img = img.reshape(expected_shape)

            return img
        except Exception as e:
            print(f"Warning: Failed to load image {im_path}: {e}")
            return None

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
    
    ## SW: Split images by tile ID to avoid data leakage
    # This function assumes that the filenames contain tile IDs in a specific format (4x4 or 3x3).
    # It splits the images into training, validation, and test sets based on these tile #.
    def split_by_tile_id(self, pos_ims, neg_ims,
                         channel='Confocal',
                         train_nums=None,
                         val_nums=None,
                         test_nums=None):
        """Split images based on tile ID patterns in filenames.

        Args:
            pos_ims:     list of positive image paths
            neg_ims:     list of negative image paths
            channel:     channel suffix in filename pattern, e.g. 'Confocal' or 'Epi'
                         Patterns are built as  _<num>_<channel>
            train_nums:  list of tile numbers assigned to training set
                         Default: [1, 4, 6, 7, 9, 10, 12, 14, 15, 16, 19, 20]
            val_nums:    list of tile numbers assigned to validation set
                         Default: [2, 8, 11, 17]
            test_nums:   list of tile numbers assigned to test set
                         Default: [3, 5, 13, 18]
        """
        if train_nums is None:
            train_nums = [1, 4, 6, 7, 9, 10, 12, 14, 15, 16, 19, 20]
        if val_nums is None:
            val_nums = [2, 8, 11, 17]
        if test_nums is None:
            test_nums = [3, 5, 13, 18]

        train_tiles = [f'_{n}_{channel}' for n in train_nums]
        val_tiles   = [f'_{n}_{channel}' for n in val_nums]
        test_tiles  = [f'_{n}_{channel}' for n in test_nums]

        print(f"DEBUG: Total images - Pos: {len(pos_ims)}, Neg: {len(neg_ims)}")
        print(f"DEBUG: channel='{channel}', train_nums={train_nums}, val_nums={val_nums}, test_nums={test_nums}")
        print(f"DEBUG: First few positive image paths:")
        for i, img in enumerate(pos_ims[:3]):
            print(f"  {i}: {img}")

        def assign_split(filename):
            for tile in train_tiles:
                if tile in filename:
                    return 'train'
            for tile in val_tiles:
                if tile in filename:
                    return 'val'
            for tile in test_tiles:
                if tile in filename:
                    return 'test'
            return None  # File doesn't match any tile pattern

        # Test the pattern matching with first few files
        print(f"DEBUG: Testing pattern matching:")
        for img in pos_ims[:3]:
            result = assign_split(img)
            print(f"  {img} -> {result}")

        # Split positive images
        pos_train, pos_val, pos_test = [], [], []
        unmatched_pos = []
        for img in pos_ims:
            split = assign_split(img)
            if split == 'train':
                pos_train.append(img)
            elif split == 'val':
                pos_val.append(img)
            elif split == 'test':
                pos_test.append(img)
            else:
                unmatched_pos.append(img)

        # Split negative images
        neg_train, neg_val, neg_test = [], [], []
        unmatched_neg = []
        for img in neg_ims:
            split = assign_split(img)
            if split == 'train':
                neg_train.append(img)
            elif split == 'val':
                neg_val.append(img)
            elif split == 'test':
                neg_test.append(img)
            else:
                unmatched_neg.append(img)

        print(f"Tile-based split results:")
        print(f"Train: {len(pos_train)} pos, {len(neg_train)} neg")
        print(f"Val:   {len(pos_val)} pos, {len(neg_val)} neg")
        print(f"Test:  {len(pos_test)} pos, {len(neg_test)} neg")
        print(f"Unmatched: {len(unmatched_pos)} pos, {len(unmatched_neg)} neg")

        if unmatched_pos:
            print(f"Sample unmatched positive examples:")
            for img in unmatched_pos[:3]:
                print(f"  {img}")

        if len(pos_val) == 0 and len(neg_val) == 0:
            print("WARNING: No validation images found!")

        return pos_train, pos_val, pos_test, neg_train, neg_val, neg_test
 

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
                print('Cutoff dataset: smaller: {}, larger: {}'.format(len(lista), len(listb)))
                print('Cutoff dataset: new smaller: {}, new larger: {}'.format(len(big_new), len(listb)))
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

                # Skip corrupted/empty files
                if img is None:
                    continue

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


class MulticlassRecord(Record):
    """TFRecord builder for N-class classification.

    Args:
        class_img_lists: list of image-path lists, one per class.
                         Label for each class = its index in the list.
        tfrecord_dir:    directory to write tfrecords
        split:           [train_frac, val_frac, test_frac]
        balance_method:  'cutoff' | 'multiply' | None
    """

    def __init__(self, class_img_lists, tfrecord_dir, split, balance_method='cutoff'):
        self.p = param.Param()
        self.tfrecord_dir = tfrecord_dir
        self.balance_method = balance_method
        self.num_classes = len(class_img_lists)

        # Optional: cutoff all classes to the size of the smallest class
        if balance_method == 'cutoff':
            min_len = min(len(imgs) for imgs in class_img_lists)
            class_img_lists = [random.sample(imgs, min_len) for imgs in class_img_lists]

        all_train, all_val, all_test = [], [], []
        all_train_lbls, all_val_lbls, all_test_lbls = [], [], []

        for label, img_list in enumerate(class_img_lists):
            n = len(img_list)
            train = img_list[:int(n * split[0])]
            val = img_list[int(n * split[0]):int(n * (split[0] + split[1]))]
            test = img_list[int(n * (split[0] + split[1])):]
            all_train.extend(train)
            all_val.extend(val)
            all_test.extend(test)
            all_train_lbls.extend([label] * len(train))
            all_val_lbls.extend([label] * len(val))
            all_test_lbls.extend([label] * len(test))
            print(f'Class {label}: {len(train)} train, {len(val)} val, {len(test)} test')

        self.trainpaths, self.trainlbls = self._shuffle(all_train, all_train_lbls)
        self.valpaths, self.vallbls = self._shuffle(all_val, all_val_lbls)
        self.testpaths, self.testlbls = self._shuffle(all_test, all_test_lbls)

    def _shuffle(self, paths, lbls):
        paths = np.array(paths)
        lbls = np.int16(np.array(lbls))
        idx = np.arange(len(paths))
        np.random.seed(0)
        np.random.shuffle(idx)
        return paths[idx], lbls[idx]


if __name__ == '__main__':
    p = param.Param()
    pos_dir = '/gladstone/finkbeiner/linsley/Shijie_ML/TauKO/Mito/'
    neg_dir = '/gladstone/finkbeiner/linsley/Shijie_ML/TauWT/Mito/'
    split = [.7, .15, .15]
    img_exts = ('*.png', '*.tif', '*.tiff')
    pos_ims = []
    neg_ims = []
    for ext in img_exts:
        pos_ims.extend(glob.glob(os.path.join(pos_dir, ext)))
        neg_ims.extend(glob.glob(os.path.join(neg_dir, ext)))

    # poss = [f for f in _poss if get_timepoint(f) < 11]
    # negs = [f for f in _negs if get_timepoint(f) < 11]
    print(f"Found {len(pos_ims)} positive image candidates")
    print(f"Found {len(neg_ims)} negative image candidates")

    # SW: removes the timepoint filter
    # poss = [f for f in pos_ims if get_timepoint(f) < 11]
    # negs = [f for f in neg_ims if get_timepoint(f) < 11]
    poss = pos_ims
    negs = neg_ims

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
