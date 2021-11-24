"""
Model class.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import param_gedi as param
import numpy as np


class CNN:
    def __init__(self, trainable=True):
        self.p = param.Param()
        self.trainable = trainable

    def custom_model(self, imsize):
        act = 'relu'
        inputs = tf.keras.Input(shape=imsize, name='input_1')
        x = layers.Conv2D(32, (3, 3), activation=act)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation=act)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation=act)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(64)(x)
        x = layers.Dense(self.p.output_size, activation='softmax', name='output')(x)

        raw_model = tf.keras.Model(inputs=inputs, outputs=x)
        raw_model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)
        # optimizer = tfa.optimizers.AdamW(learning_rate=self.p.learning_rate, weight_decay=self.p.wd)
        raw_model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        return raw_model

    def custom_model2(self, imsize):
        act = 'selu'
        switch = 'instance'

        def bn(x):
            return layers.BatchNormalization()(x)

        def instance(x):
            return tfa.layers.InstanceNormalization(axis=3,
                                                    center=True,
                                                    scale=True,
                                                    beta_initializer="random_uniform",
                                                    gamma_initializer="random_uniform")(x)

        def identity(x):
            return x

        if switch == 'bn':
            norm = bn
        elif switch == 'instance':
            norm = instance
        else:
            norm = identity

        # norm = bn if switch == 'bn' else instance
        inputs = tf.keras.Input(shape=(imsize), name='input_1')
        x = layers.Conv2D(64, (3, 3), activation=act)(inputs)
        x = norm(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation=act)(x)
        x = norm(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation=act)(x)
        x = norm(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation=act)(x)
        x = norm(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation=act)(x)
        x = norm(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128)(x)
        x = layers.Dense(self.p.output_size, activation='softmax')(x)

        raw_model = tf.keras.Model(inputs=inputs, outputs=x)
        raw_model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)
        # optimizer = tfa.optimizers.AdamW(learning_rate=self.p.learning_rate, weight_decay=self.p.wd)
        raw_model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        return raw_model

    def _custom_model(self, imsize):
        """
        Model with a few conv layers, no transfer learning
        Args:
            imsize:

        Returns:

        """
        head = tf.keras.Sequential()
        head.add(layers.Conv2D(32, (3, 3), input_shape=(imsize[0], imsize[1], imsize[2])))
        head.add(layers.BatchNormalization())
        head.add(layers.Activation('relu'))
        head.add(layers.MaxPooling2D(pool_size=(2, 2)))
        head.add(layers.Conv2D(32, (3, 3)))
        head.add(layers.BatchNormalization())
        head.add(layers.Activation('relu'))
        head.add(layers.MaxPooling2D(pool_size=(2, 2)))
        head.add(layers.Conv2D(64, (3, 3)))
        head.add(layers.BatchNormalization())
        head.add(layers.Activation('relu'))
        head.add(layers.MaxPooling2D(pool_size=(2, 2)))
        average_pool = tf.keras.Sequential()
        average_pool.add(layers.AveragePooling2D())
        average_pool.add(layers.Flatten())
        average_pool.add(layers.Dense(self.p.output_size, activation='sigmoid'))
        raw_model = tf.keras.Sequential([
            head,
            average_pool
        ])
        raw_model.summary()
        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def dropout(self, name):
        if self.trainable:
            return layers.Dropout(rate=0.5, name=name)
        else:
            return layers.Dropout(rate=0.5, seed=11, name=name)

    def vgg16(self, imsize=(224, 224, 3), batchnorm=False):
        """
        VGG16 model. Based on model in original tf 1.x gedi repo.
        Args:
            imsize: Image size
            batchnorm: Boolean to use batch norm or not

        Returns:
            model: compiled model
        """
        input = layers.Input(shape=(imsize[0], imsize[1], imsize[2]), name='input_1')  # NAME MATCHES DICT KEY
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input,
                                                 input_shape=(imsize[0], imsize[1], imsize[2]))
        for layr in base_model.layers:
            # if (('block4' in layr.name) or ('block5' in layr.name)):
            if ('block5' in layr.name):
                # if 'block5' in layr.name:

                layr.trainable = True
            else:
                layr.trainable = False
        glorot = tf.initializers.GlorotUniform()
        for layr in base_model.layers:
            # if (('block4_conv' in layr.name) or ('block5_conv' in layr.name)):
            if ('block5_conv' in layr.name):
                _weights = layr.get_weights()
                W = np.shape(_weights[0])
                b = np.shape(_weights[1])
                layr.set_weights([glorot(shape=W), glorot(shape=b)])

        drop1 = layers.Dropout(rate=0.5, name='dropout_1')
        drop2 = layers.Dropout(rate=0.5, name='dropout_2')
        drop3 = layers.Dropout(rate=0.5, name='dropout_3')
        bn1 = layers.BatchNormalization(momentum=0.9, name='bn_1')
        bn2 = layers.BatchNormalization(momentum=0.9, name='bn_2')
        bn3 = layers.BatchNormalization(momentum=0.9, name='bn_3')
        # instance1 = tfa.layers.InstanceNormalization(name='instance_1')
        # instance2 = tfa.layers.InstanceNormalization(name='instance_2')
        # updated_model = tf.keras.models.Sequential()
        # for layer in base_model.layers:
        #     updated_model.add(layer)
        #     if 'block5_conv' in layer.name:
        #         num = layer.name[-1]
        #         updated_model.add(layers.BatchNormalization(name='batch_normalization_{}'.format(num)))

        conv5_1 = base_model.get_layer('block5_conv1')
        conv5_2 = base_model.get_layer('block5_conv2')
        conv5_3 = base_model.get_layer('block5_conv3')
        block5_pool = base_model.get_layer('block5_pool')

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        flatten = tf.keras.layers.Flatten()

        fc1 = layers.Dense(4096, name='dense_1', activation='relu', kernel_initializer='TruncatedNormal',
                           bias_initializer='TruncatedNormal')
        fc2 = layers.Dense(4096, name='dense_2', activation='relu', kernel_initializer='TruncatedNormal',
                           bias_initializer='TruncatedNormal')
        # fc3 = layers.Dense(256, activation='relu', name='dense_3')
        fc3 = layers.Dense(self.p.output_size, name='fc3')

        # updated_model.summary()
        # x = updated_model(input)
        # x = global_average_layer(x)
        x = flatten(block5_pool.output)
        x = fc1(x)
        x = bn1(x)
        # x = drop1(x, training=self.trainable)
        # x = instance1(x)
        x = fc2(x)
        x = bn2(x)
        # x = drop2(x, training=self.trainable)
        # x = instance2(x)
        x = fc3(x)
        x = tf.keras.layers.Softmax(name='output')(x)

        raw_model = Model(inputs=base_model.input, outputs=x)
        raw_model.summary()

        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def vgg19(self, imsize):
        initializer = 'TruncatedNormal'
        # initializer = 'glorot_uniform'
        base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                                 input_shape=(imsize[0], imsize[1], imsize[2]))
        # base_model.trainable = False
        flatten = tf.keras.layers.Flatten()
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        bn1 = layers.BatchNormalization(momentum=0.9, name='bn_1')
        bn2 = layers.BatchNormalization(momentum=0.9, name='bn_2')
        bn3 = layers.BatchNormalization(momentum=0.9, name='bn_3')
        fc1 = layers.Dense(256, kernel_initializer=initializer, bias_initializer=initializer, activation='relu',
                           name='dense_1')
        fc2 = layers.Dense(256, kernel_initializer=initializer, bias_initializer=initializer, activation='relu',
                           name='dense_2')
        fc3 = layers.Dense(self.p.output_size, name='fc3')
        drop1 = layers.Dropout(rate=0.5, name='dropout_1')
        drop2 = layers.Dropout(rate=0.5, name='dropout_2')
        drop3 = layers.Dropout(rate=0.5, name='dropout_3')
        prediction = layers.Dense(self.p.output_size, activation='softmax', name='output')
        block5_pool = base_model.get_layer('block5_pool')
        for layr in base_model.layers:
            # ininy = tf.initializers.GlorotUniform()
            ininy = tf.initializers.TruncatedNormal()

            if ('block5' in layr.name) or ('block4' in layr.name):
                # if ('block5' in layr.name):
                # _weights = layr.get_weights()
                # if len(_weights) > 0:
                #     # print('resetting weights:', layr.name)
                #     W = np.shape(_weights[0])
                #     b = np.shape(_weights[1])
                #     layr.set_weights([ininy(shape=W), ininy(shape=b)])

                layr.trainable = True
            else:
                layr.trainable = False
            print(layr.trainable)

        x = flatten(block5_pool.output)
        x = fc1(x)
        # x = bn1(x)
        # x = drop1(x, training=self.trainable)
        # x = instance1(x)
        x = fc2(x)
        # x = bn2(x)
        # x = drop2(x, training=self.trainable)
        # x = instance2(x)
        x = fc3(x)
        x = tf.keras.layers.Softmax(name='output')(x)

        raw_model = Model(inputs=base_model.input, outputs=x)
        raw_model.summary()
        if self.p.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)
        elif self.p.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.p.learning_rate, momentum=self.p.momentum,
                                                nesterov=True)

        raw_model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def inceptionv3(self, imsize=(299, 299, 3)):
        """
        Input shape must be greater than (75,75,3), (299,299,3) is default.
        :return:
        """
        base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                    input_shape=(imsize[0], imsize[1], imsize[2]))
        for layr in base_model.layers[:100]:
            layr.trainable = False
        for layr in base_model.layers[100:]:
            layr.trainable = True
        base_model.summary()
        flat = layers.Flatten()
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        fc1 = layers.Dense(16, activation='relu', name='dense_1')
        fc2 = layers.Dense(16, activation='relu', name='dense_2')
        fc3 = layers.Dense(128, activation='softmax', name='dense_3')
        prediction = layers.Dense(self.p.output_size, activation='softmax', name='output')
        raw_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            fc1,
            fc2,
            prediction
        ])
        raw_model.summary()

        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def resnet50(self, imsize):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                                    input_shape=(imsize[0], imsize[1], imsize[2]))
        base_model.trainable = False
        fc1 = layers.Dense(128, activation='relu', name='dense_1')
        fc2 = layers.Dense(256, activation='relu', name='dense_2')
        fc3 = layers.Dense(256, activation='relu', name='dense_3')
        global_pool = layers.GlobalAveragePooling2D()
        prediction = layers.Dense(self.p.output_size, name='prediction')
        flat = layers.Flatten()
        conv_lyr = base_model.get_layer('conv5_block3_out')
        base_model.summary()
        for layr in base_model.layers:
            if ('conv4' in layr.name) or ('conv5' in layr.name):

                layr.trainable = True
            else:
                layr.trainable = False
            print(layr.trainable)

        x = global_pool(conv_lyr.output)
        x = prediction(x)
        x = layers.Softmax(name='output')(x)
        raw_model = Model(inputs=base_model.input, outputs=x)
        raw_model.summary()

        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def mobilenet(self, imsize):
        base_model = tf.keras.applications.MobileNetV2(input_shape=(imsize[0], imsize[1], imsize[2]),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        prediction_layer = layers.Dense(self.p.output_size)

        raw_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        raw_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.p.learning_rate),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
        raw_model.summary()
        return raw_model


if __name__ == '__main__':
    Net = CNN()
    model = Net.vgg16()
