import os
import platform

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import param_gedi as param


class GradOps:
    def __init__(self,p,  vgg_normalize=True):
        os_type = platform.system()
        self.p = p
        if os_type == 'Darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac has KMP issues

        # File locations

        self.tfrec_dir = {
            'Darwin': '/Volumes/data/robodata/Gennadi/gedi_data/tfrecs'
            , 'Linux': '/mnt/data/gedi/activationmap/tfrecs'
        }[os_type]

        self.ckpt_dir = {
            'Darwin': '/Volumes/data/robodata/Gennadi/gedi_data/ckpts'
            , 'Linux': '/mnt/data/gedi/activationmap/ckpts'
        }[os_type]

        self.log_dir = {
            'Darwin': '/Volumes/data/robodata/Gennadi/gedi_data/logs'
            , 'Linux': '/mnt/data/gedi/activationmap/logs'
        }[os_type]

        self.train_rec = os.path.join(self.tfrec_dir,
                                      'all_data_train.tfrecords')  # 187772 images {'dead': 89, 'live': 279}
        self.val_rec = os.path.join(self.tfrec_dir, 'all_data_val.tfrecords')  # 20863 images {'dead': 16, 'live': 27}

        self.train_rec_size = 187772
        self.val_rec_size = 20863

        self.class_weights = {0: 3., 1: 1.}  # rough ratio

        # # Obsolete; whole dataset used including remainder batch
        # self.train_data_skip = self.train_rec_size % self.batcher_params['batch_size']
        # self.val_data_skip = self.val_rec_size % self.batcher_params['batch_size']

        # Data transforms/specs

        self.batcher_params = {
            'batch_size': 16
            , 'orig_size': (300, 300)
            , 'target_size': (224, 224)
            , 'crop_type': 'center'
            , 'parallel_reads': None  # possibly useful; not applied so far
        }

        self.train_transform_params = {
            'rotation_range': 90
            , 'width_shift_range': 0.2
            , 'height_shift_range': 0.2
            , 'shear_range': 0.2
            , 'zoom_range': 0.2
            , 'horizontal_flip': True
            , 'vertical_flip': True
            , 'fill_mode': 'nearest'
        }

        self.mean_bgr = self.p.VGG_MEAN

        self.vgg_normalize = vgg_normalize

        # Training specs

        self.train_epoch_steps = self.train_rec_size // self.batcher_params['batch_size'] + int(bool(
            self.train_rec_size % self.batcher_params[
                'batch_size']))  # add one more step if there is a leftover small batch
        self.val_epoch_steps = self.val_rec_size // self.batcher_params['batch_size'] + int(
            bool(self.val_rec_size % self.batcher_params['batch_size']))

    def tfrec_parse(self, row):
        """
        Abstracts away tfrecord formats

        Args:
            row:

        Returns:

        """

        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'ratio': tf.FixedLenFeature([], tf.float32)
        }
        parsed = tf.parse_single_example(row, features)
        parsed['image'] = tf.decode_raw(parsed['image'], tf.float32)

        return parsed['image'], parsed['label']

    # def img_parse(self, img, transformer=None):
    #     orig_size, target_size, crop_type = self.batcher_params['orig_size'], self.batcher_params['target_size'], self.batcher_params['crop_type']
    #
    #     img = np.array(img).reshape(orig_size)  # Make 2d
    #
    #     # Crop to fit target size
    #     delta = np.array([a - b for a, b in zip(orig_size, target_size)])
    #     assert min(delta) >= 0, 'Cannot crop images to be larger' # we are not trying to make image bigger
    #     if min(delta) > 0: # we want to crop image
    #         # Find pixel for crop start
    #         if crop_type == 'random':
    #             delta = np.array(list(map(lambda x: random.randint(0, x), delta)))
    #         else:
    #             # Defaults to center crop
    #             delta //= 2
    #         img = img[delta[0]:delta[0] + target_size[0], delta[1]:delta[1] + target_size[1]]
    #
    #     # Triplicate channels
    #     img = np.repeat(img, 3, -1).reshape(target_size + (3,))
    #
    #     # Apply custom augments if applicable
    #     if transformer is not None:
    #         img = transformer.random_transform(img)
    #
    #     # Normalize
    #     img -= np.amin(img)
    #     img /= np.amax(img)
    #
    #     return img

    def img_parse(self, img, transformer=None, rgb=True):
        img = np.array(img, dtype=np.float)

        orig_size, target_size, crop_type = img.shape[:2], self.batcher_params['target_size'], self.batcher_params[
            'crop_type']
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
            assert img.shape[-1] == 3
        if not rgb:
            img = np.repeat(np.array(img).reshape(orig_size), 3, -1).reshape(orig_size + (3,))  # Make 3d w/ channels

        # Apply custom augments if applicable
        if transformer is not None:
            img = transformer.random_transform(img)

        # Crop to fit target size
        delta = np.array([a - b for a, b in zip(orig_size, target_size)])
        assert min(delta) >= 0, 'Cannot crop images to be larger'  # we are not trying to make image bigger
        if min(delta) > 0:  # we want to crop image
            # Find pixel for crop start
            if crop_type == 'random':
                delta = np.array(list(map(lambda x: random.randint(0, x), delta)))
            else:
                # Defaults to center crop
                delta //= 2
            img = img[delta[0]:delta[0] + target_size[0], delta[1]:delta[1] + target_size[1], :]

        # img -= np.amin(img)  # 2020/5/18, commented to match GEDI3-master normalization
        # img /= np.amax(img)

        # img = (img - self.p.orig_min_value) / (self.p.orig_max_value - self.p.orig_min_value)
        # img = (img - self.p.training_min_value) / (self.p.training_max_value-self.p.training_min_value)

        # Normalize according to VGG specs
        if self.vgg_normalize:
            img *= 255.
            img = img[..., ::-1]  # BGR
            img[..., 0] -= self.mean_bgr[0]
            img[..., 1] -= self.mean_bgr[1]
            img[..., 2] -= self.mean_bgr[2]

        return img
