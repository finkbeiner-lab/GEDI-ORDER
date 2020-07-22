"""
Add layers to base gedi model and save
"""

import tensorflow as tf
import param_gedi as param
from models.model import CNN
import preprocessing.datagenerator as pipe
from utils.utils import update_timestring, make_directories
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

savename = '/mnt/finkbeinerlab/robodata/GEDI_CLUSTER/base_gedi_dropout2.h5'

p = param.Param()


print('Loading model...')
base_model = tf.keras.models.load_model(p.base_gedi, compile=False)
# visualize kernels to check model weights
# if rapid chagne and plateu check data, biases chenged with grad descent
# check base model included in graph and weigths are  changing
# Check that live = 1, dead = 0
# verify that new data live/dead look accurate
# try different learning rates
# vgg16 lesion some layers with random weights, initialize with imagenet, train with 10000 images
# Mismatch between labels and data, dead=dead, live = live
# no black images
# contrast range is similar, contrast normalization skimage, histogram equalization.
# histogram matching of old vs new data histogram normalization
# represent every image by mean intensity
# learn affine transformation, scale and intercept, on average transform human to rat

glorot = tf.initializers.GlorotUniform()
bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_1')
bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_2')
bn3 = tf.keras.layers.BatchNormalization(momentum=0.9, name='bn_3')
# fc1_small = tf.keras.layers.Dense(64, name='fc1', activation='relu', kernel_initializer='TruncatedNormal',
#                    bias_initializer='TruncatedNormal')
# fc2_small = tf.keras.layers.Dense(16, name='fc2', activation='relu', kernel_initializer='TruncatedNormal',
#                    bias_initializer='TruncatedNormal')
fc3 = tf.keras.layers.Dense(p.output_size, name='fc3')

drop1 = tf.keras.layers.Dropout(rate=0.5, name='dropout_1')
drop2 = tf.keras.layers.Dropout(rate=0.5, name='dropout_2')
block5_pool = base_model.get_layer('block5_pool')
flatten = tf.keras.layers.Flatten()

#
fc1 = base_model.get_layer('fc1')
fc2 = base_model.get_layer('fc2')

x = flatten(block5_pool.output)
x = fc1(x)
x = drop1(x)
# x = bn1(x)
x = fc2(x)
x = drop2(x)
# x = bn2(x)
x = fc3(x)
# x = bn3(x)
# x = tf.keras.layers.Softmax(name='output')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.save(savename)
