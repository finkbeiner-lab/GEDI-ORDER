"""
Gradcam in tf.20 - BROKEN, currently tf2.0 doesn't grab layers like in tf1.x. Use store_viz_stacks.py.
"""

import tensorflow as tf
from ops.gradcam_ops import Gradcam
import preprocessing.datagenerator as pipe
import param_gedi as param
import os
import numpy as np

p = param.Param()
layer_name = 'block5_conv3'
# layer_name = 'block1_conv1'
# layer_name = 'bn_3'
tfrecord = p.data_deploy
DEBUG = True
gen = pipe.Dataspring(tfrecord)
gen.datagen_base(istraining=True)
timestamp = 'vgg16_2020_04_17_15_43_26'
import_path = os.path.join(p.models_dir, "{}.h5".format(timestamp))
# import_path = os.path.join(p.ckpt_dir, "{}.hdf5".format(timestamp))

print('Loading model...')
model = tf.keras.models.load_model(import_path, compile=False)
for lyr in model.layers:
    lyr.trainable=False
model.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model.summary()
# print([layer.output.name for layer in model.get_layer('sequential').layers])
gcam = Gradcam(model, layer_name=layer_name, debug=DEBUG)
for i in range(10):
    # image_batch, lbl_batch = DatTest.datagen()
    # # Plops.show_batch(image_batch, lbl_batch)

    imgs, lbls, files = gen.datagen()
    # imgs = np.array(imgs)
    # nplbls = lbls.numpy()
    # inputimg = np.reshape(imgs[0], (1, 224, 224, 3))
    # inputlbl = np.reshape(nplbls[0], (1, 2))
    gcam.guided_backprop(imgs, lbls)
