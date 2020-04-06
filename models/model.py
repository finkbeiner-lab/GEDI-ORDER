import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ops.model_ops import insert_layer_nonseq
import param_gedi as param


class CNN:
    def __init__(self, trainable=True):
        self.p = param.Param()
        self.trainable = trainable

    def standard_model(self, imsize):
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

    def vgg16(self, imsize=(224, 224, 3)):
        input = layers.Input(shape=(imsize[0], imsize[1], imsize[2]), name='imgs')
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input,
                                                 input_shape=(imsize[0], imsize[1], imsize[2]))
        for layr in base_model.layers:
            if (('block4' in layr.name) or ('block5' in layr.name)):
                # if 'block5' in layr.name:

                layr.trainable = True
            else:
                layr.trainable = False
        drop1 = layers.Dropout(rate=0.5, name='dropout_1')
        drop2 = layers.Dropout(rate=0.5, name='dropout_2')
        drop3 = layers.Dropout(rate=0.5, name='dropout_3')
        bn1 = layers.BatchNormalization(name='bn_1')
        bn2 = layers.BatchNormalization(name='bn_2')
        bn3 = layers.BatchNormalization(name='bn_3')
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
        y = bn1(conv5_1.output)
        y = conv5_2(y)
        y = bn2(y)
        y = conv5_3(y)
        y = bn3(y)
        x = block5_pool(y)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        fc1 = layers.Dense(256, activation='relu', name='dense_1')
        fc2 = layers.Dense(256, activation='relu', name='dense_2')
        fc3 = layers.Dense(256, activation='relu', name='dense_3')
        prediction = layers.Dense(self.p.output_size, activation='softmax', name='output')


        # updated_model.summary()
        # x = updated_model(input)
        x = global_average_layer(x)
        x = fc1(x)
        # x = bn1(x)
        x = drop1(x)
        # x = bn2(x)
        x = fc2(x)
        x = drop2(x)
        x = prediction(x)

        raw_model = Model(inputs=base_model.input, outputs=x)
        raw_model.summary()

        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return raw_model

    def xvgg16(self, imsize):
        input = layers.Input(shape=(imsize[0], imsize[1], imsize[2]), name='image_input')
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                                 input_shape=(imsize[0], imsize[1], imsize[2]))
        # base_model.trainable = False
        flat = layers.Flatten()
        dropped = layers.Dropout(0.5)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        fc1 = layers.Dense(16, activation='relu', name='dense_1')
        fc2 = layers.Dense(16, activation='relu', name='dense_2')
        fc3 = layers.Dense(128, activation='relu', name='dense_3')
        prediction = layers.Dense(self.p.output_size, activation='softmax', name='result')
        for layr in base_model.layers:
            if ('block5' in layr.name):

                layr.trainable = True
            else:
                layr.trainable = False

        base_model.summary()

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

    def vgg19(self, imsize):
        base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                                 input_shape=(imsize[0], imsize[1], imsize[2]))
        # base_model.trainable = False
        flat = layers.Flatten()
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        fc1 = layers.Dense(16, activation='relu', name='dense_1')
        fc2 = layers.Dense(16, activation='relu', name='dense_2')
        fc3 = layers.Dense(128, activation='softmax', name='dense_3')
        prediction = layers.Dense(self.p.output_size, activation='softmax', name='output')
        for layr in base_model.layers:
            if ('block5' in layr.name):

                layr.trainable = True
            else:
                layr.trainable = False
            print(layr.trainable)
        raw_model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            fc1,
            fc2,
            prediction
        ])
        base_model.summary()

        raw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate),
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
        # todo: set the layers necessary to trainable
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
        prediction = layers.Dense(self.p.output_size, name='output')
        flat = layers.Flatten()
        for layr in base_model.layers:
            if ('conv4' in layr.name) or ('conv5' in layr.name):

                layr.trainable = True
            else:
                layr.trainable = False
            print(layr.trainable)
        raw_model = tf.keras.Sequential([
            base_model,
            flat,
            fc1,
            fc2,
            prediction
        ])
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
