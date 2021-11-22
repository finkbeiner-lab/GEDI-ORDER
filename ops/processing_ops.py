import numpy as np
import tensorflow as tf
import random
import param_gedi as param


class Parser:
    """Parse tfrecord"""
    def __init__(self):
        self.p = param.Param()

    def tfrec_parse(self, row):
        """
        Parse single item in tfrecord. You most likely want tfrec_batch_parse.

        Args:
            row:  A scalar string Tensor, a single serialized Example.

        Returns:

        """

        features = {
            # 'filename': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(row, features)
        # file = parsed['filename']
        img = tf.decode_raw(parsed['image'], tf.float32)
        lbl = tf.cast(parsed['label'], tf.int32)
        lbls = tf.one_hot(lbl, 2)  # test this in pytest
        img = tf.decode_raw(img, tf.float32)
        img = tf.reshape(img, [-1, self.p.orig_width, self.p.orig_height, self.p.orig_channels])
        if self.p.orig_height > self.p.vgg_height:
            x0 = (self.p.orig_width - self.p.vgg_width) // 2
            y0 = (self.p.orig_height - self.p.vgg_height) // 2
            img = tf.image.crop_to_bounding_box(img, y0, x0, 224, 224)
        # img = tf.divide(img, 255.0)  # normalize here
        # img = tf.divide(img, self.p.max_gedi)  # normalize here

        return img, lbls

    def tfrec_batch_parse(self, row):
        """
        Parse tfrecord by batch.

        Args:
            row: A scalar string Tensor, a single serialized Example.

        Returns:

        """

        features = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'ratio': tf.io.FixedLenFeature([], tf.float32)
        }


        parsed = tf.io.parse_example(row, features)
        files = parsed['filename']
        img = tf.io.decode_raw(parsed['image'], tf.float32)
        lbl = tf.cast(parsed['label'], tf.int32)
        ratio = parsed['ratio']  # useful to have ratio, but model is a binary classifier with binary ground truth.
        lbls = tf.one_hot(lbl, 2)  # one hot, verify this in pytest
        img = tf.reshape(img, [-1, self.p.orig_size[0], self.p.orig_size[1], self.p.orig_size[2]])
        # if self.p.orig_size[0] > self.p.target_size[0]:
        #     x0 = (self.p.orig_size[1] - self.p.target_size[1]) // 2
        #     y0 = (self.p.orig_size[0] - self.p.target_size[0]) // 2
        #     img = tf.image.crop_to_bounding_box(img, y0, x0, 224, 224)
        # img = tf.divide(img, self.p.max_gedi)  # normalize here

        return img, lbls, files

    def reshape_ims(self, imgs, lbls, files):
        """
        Reshape images for model
        """
        if self.p.orig_size[-1] > self.p.target_size[-1]:
            # Remove alpha channel
            channels = tf.unstack(imgs, axis=-1)
            imgs = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
        if self.p.randomcrop:
            if self.p.orig_size[0] > self.p.target_size[0]:
                imgs = tf.image.random_crop(imgs, size=[self.p.BATCH_SIZE, 224, 224, 1])
        else:
            if self.p.orig_size[0] > self.p.target_size[0]:
                y0 = (self.p.orig_size[0] - self.p.target_size[0]) // 2
                x0 = (self.p.orig_size[1] - self.p.target_size[1]) // 2

                imgs = tf.image.crop_to_bounding_box(imgs, y0, x0, self.p.target_size[1], self.p.target_size[0])
        return imgs, lbls, files

    @staticmethod
    def transformImg(imgIn, forward_transform):
        """
        https://stackoverflow.com/questions/52214953/tensorflow-is-there-a-way-to-implement-tensor-wise-image-shear-rotation-transl
        Args:
            imgIn:
            forward_transform:

        Returns:

        """
        t = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
        # please notice that forward_transform must be a float matrix,
        # e.g. [[2.0,0,0],[0,1.0,0],[0,0,1]] will work
        # but [[2,0,0],[0,1,0],[0,0,1]] will not
        imgOut = tf.contrib.image.transform(imgIn, t, interpolation="BILINEAR", name=None)
        return imgOut

    def use_binary_lbls(self, imgs, lbls, files):
        """Convert from one-hot encoded labels to single digit label 0 or 1"""
        newlbls = tf.argmax(lbls, axis=1)
        newlbls = tf.cast(newlbls, dtype=tf.float32)
        return imgs, newlbls, files

    def inception_scale(self, imgs, lbls, files):
        """Scaling for inception model"""
        assert_op = tf.Assert(tf.less_equal(tf.reduce_max(imgs), 1.0), [imgs])
        with tf.control_dependencies([assert_op]):
            images = tf.subtract(imgs, 1.0)
            images = tf.multiply(images, 2.0)
        return images, lbls, files

    def random_shear(self, img):
        """
        Shears the whole batch the crops.
        Args:
            img:

        Returns:

        """
        # shear .5 would shear 50% of width, so it would be 112 of 224px.
        shear_x = np.random.uniform(0, 0.1)
        shear_y = np.random.uniform(0, 0.1)
        forward_transform = [[1.0, shear_x, 0],
                             [shear_y, 1.0, 0],
                             [0, 0, 1.0]]
        shear = self.transformImg(img, forward_transform)

        scales = np.ones(self.p.batch_size) * .8
        boxes = np.zeros((len(scales), 4))
        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]
        idx = [i for i in range(self.p.batch_size)]

        crops = tf.image.crop_and_resize(shear, boxes=boxes, box_ind=idx, crop_size=[224, 224])
        return crops

    def zoom(self, img, be_random=True):
        """
        Random zooms
        Args:
            img:
            thresh:
            be_random:

        Returns:

        """
        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        # scales = list(np.arange(0.8, 1.0, 0.01))
        if be_random:
            scales = list(np.random.uniform(0.8, 1.0, self.p.batch_size))
        else:
            scales = np.ones(self.p.batch_size) * .5
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]
        idx = [i for i in range(self.p.batch_size)]
        crops = tf.image.crop_and_resize(img, boxes=boxes, box_ind=idx, crop_size=[224, 224])

        # def random_crop(_img):
        #     # Create different crops for an image
        #     crops = tf.image.crop_and_resize(_img, boxes=boxes, box_ind=idx, crop_size=[224, 224])
        #     # Return a random crop
        #     # return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
        #     return crops

        # choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        #
        # # Only apply cropping 50% of the time
        #
        # # img = tf.cond(choice < thresh, lambda: img, lambda: random_crop(img))
        return crops

    def augment(self, img, lbls, files):
        """
        Augmentations. Img is set to have a max of one.
        Args:
            img:
            lbls:

        Returns:

        """
        assert_op = tf.Assert(tf.less_equal(tf.reduce_max(img), 1.0), [img])
        with tf.control_dependencies([assert_op]):

            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_flip_left_right(img)

            #
            turns = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

            img = tf.cond(tf.equal(turns, tf.constant(1)), lambda: tf.image.rot90(img, 1), lambda: tf.identity(img))

            img = tf.cond(tf.equal(turns, tf.constant(2)), lambda: tf.image.rot90(img, 2), lambda: tf.identity(img))

            img = tf.cond(tf.equal(turns, tf.constant(3)), lambda: tf.image.rot90(img, 3), lambda: tf.identity(img))

        # tensorf object has no attribute ndim error
        # img = tf.keras.preprocessing.image.random_shear(img, 1, row_axis=1, col_axis=2, channel_axis=3)
        # tf.print('max img', tf.reduce_max(img))
        # tf.print('min img', tf.reduce_min(img))
        img = tf.image.random_brightness(img,
                                         max_delta=self.p.random_brightness)  # Image normalized to 1, delta is amount of brightness to add/subtract
        img = tf.image.random_contrast(img, self.p.min_contrast, self.p.max_contrast)  # (x- mean) * contrast factor + mean
        # tf.print('aug img', tf.reduce_max(img))
        # tf.print('aug img', tf.reduce_min(img))
        return img, lbls, files

    def remove_negatives_in_img(self, img):
        """If there are negatives in image, subtract by min to get make all values nonnegative"""
        _min = tf.reduce_min(img)
        # _mins = tf.reduce_min(img, axis=0, keepdims=True)
        # _mins = tf.reduce_min(_mins, axis=1, keepdims=True)
        # _mins = tf.reduce_min(_mins, axis=2, keepdims=True)
        # _zeros = tf.zeros_like(_mins)
        # subtr = tf.where(tf.less(_mins, _zeros), _mins, _zeros)
        # img = img - subtr
        subtr = tf.where(tf.less(_min, 0.), _min, 0.)
        img = img - subtr

        return img

    def divide_by_max_in_img(self, img):
        """Divide by max in image by batch"""
        _maxs = tf.reduce_max(img, axis=0, keepdims=True)
        _maxs = tf.reduce_max(_maxs, axis=1, keepdims=True)
        _maxs = tf.reduce_max(_maxs, axis=2, keepdims=True)
        img = img / _maxs

        return img

    def cut_off_vals(self, imgs, lbls, files):
        """Cut off values"""
        imgs = tf.clip_by_value(imgs, clip_value_min=0., clip_value_max=1.)
        return imgs, lbls, files

    def normalize_whitening(self, imgs, lbls):
        """Scales each image in batch to have mean 0 and variance 1"""
        # imgs = tf.map_fn(lambda x: tf.image.per_image_standardization(x), imgs)
        imgs = tf.image.per_image_standardization(imgs)

        return imgs, lbls

    def format_example(self, imgs, lbls, files):
        assert_op = tf.Assert(tf.less_equal(tf.reduce_max(imgs), 1.0), [imgs])
        with tf.control_dependencies([assert_op]):
            images = tf.cast(imgs, tf.float32)
            images = images * 255.0
            images = (images / 127.5) - 1.0
            images = tf.image.resize(images, (self.p.target_size[0], self.p.target_size[1]))
        return images, lbls, files

    def set_max_to_one_by_image(self, imgs, lbls, files):
        """Divide each image by its maximum"""
        imgs = tf.map_fn(lambda x: self.remove_negatives_in_img(x), imgs)
        imgs = tf.map_fn(lambda x: self.divide_by_max_in_img(x), imgs)
        return imgs, lbls, files

    def set_max_to_one_by_batch(self, imgs, lbls, files):
        """Divide each batch by its maximum"""
        imgs = tf.map_fn(lambda x: self.remove_negatives_in_img(x), imgs)
        imgs = imgs / tf.reduce_max(imgs, keepdims=True)
        return imgs, lbls, files

    def rescale_im_and_clip_16bit(self, imgs, lbls, files):
        imgs = tf.map_fn(lambda x: (x - self.p.orig_min_value) / (self.p.orig_max_value - self.p.orig_min_value), imgs)
        imgs = tf.clip_by_value(imgs, clip_value_min=0., clip_value_max=1.)
        return imgs, lbls, files

    def rescale_im_and_clip_renorm(self, imgs, lbls, files):
        imgs = tf.map_fn(lambda x: (x - self.p.training_min_value) / (self.p.training_max_value - self.p.training_min_value), imgs)
        imgs = tf.clip_by_value(imgs, clip_value_min=0., clip_value_max=1.)
        return imgs, lbls, files


    def normalize_resnet(self, img, lbls, files):
        """Set up resnet"""
        rgb = img
        if int(rgb.get_shape()[-1]) == 1:
            red, green, blue = rgb, rgb, rgb
        else:
            red, green, blue = tf.split(
                axis=3, num_or_size_splits=3, value=rgb)

        assert_op = tf.Assert(tf.reduce_all(tf.equal(red.get_shape()[1:], tf.constant([224, 224, 1]))), [red])
        with tf.control_dependencies([assert_op]):
            new_img = tf.concat(axis=3, values =[
                red, green, blue
            ], name='rgb')

            # normed = tf.image.per_image_standardization(new_img)

        return new_img, lbls, files


    def make_vgg(self, img, lbls, files):
        """
        Subtracts the vgg16 training mean by channel.
        Args:
            img:
            lbls:
            files:

        Returns:

        """
        rgb = img * 255.0
        if int(img.get_shape()[-1]) == 1:
            red, green, blue = rgb, rgb, rgb
        else:
            red, green, blue = tf.split(
                axis=3, num_or_size_splits=3, value=rgb)

        assert_op = tf.Assert(tf.reduce_all(tf.equal(red.get_shape()[1:], tf.constant([224, 224, 1]))), [red])
        with tf.control_dependencies([assert_op]):
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(axis=3, values=[
                blue - self.p.VGG_MEAN[0],
                green - self.p.VGG_MEAN[1],
                red - self.p.VGG_MEAN[2],
            ], name='bgr')
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        return bgr, lbls, files

    def chk_make_vgg(self, img, lbls, files):
        """
        Subtracts the vgg16 training mean by channel.
        Args:
            img:
            lbls:
            files:

        Returns:

        """
        if int(img.get_shape()[-1]) == 1:
            red, green, blue = rgb, rgb, rgb
        else:
            red, green, blue = tf.split(
                axis=3, num_or_size_splits=3, value=rgb)

        assert_op = tf.Assert(tf.reduce_all(tf.equal(red.get_shape()[1:], tf.constant([224, 224, 1]))), [red])
        with tf.control_dependencies([assert_op]):
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(axis=3, values=[
                blue - self.p.VGG_MEAN[0],
                green - self.p.VGG_MEAN[1],
                red - self.p.VGG_MEAN[2],
            ], name='bgr')
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        return bgr, lbls, files


if __name__=='__main__':
    p = param.Param()
    Parse = Parser()
    # get_tfrecord_length(p.train_rec, get_max=True)
    ds = tf.data.TFRecordDataset(p.data_test,
                                 num_parallel_reads=p.num_parallel_calls)  # possibly use multiple record files
    ds = ds.batch(p.BATCH_SIZE, drop_remainder=False)  # batch images, no skips
    ds = ds.map(Parse.tfrec_batch_parse,
                num_parallel_calls=p.num_parallel_calls)
    it = iter(ds)
    img, lbls, files, ratio = next(it)
    print(ratio)