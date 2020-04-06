import tensorflow as tf
from ops.gradcam_ops import Gradcam
import preprocessing.datagenerator as pipe
import param_gedi as param
import os
import numpy as np
import matplotlib.pyplot as plt

p = param.Param()
timestamp = 'vgg16_2020_03_17_15_34_30'
import_path = os.path.join(p.models_dir, "{}.h5".format(timestamp))

tfrecord = p.data_train
DEBUG = True

gen = pipe.Dataspring(tfrecord)
gen.datagen_base(istraining=False)
layer_name = 'block5_conv3'
model = tf.keras.models.load_model(import_path)
model.trainable = False
model_layer = 'vgg16'
print('input')
for lyr in model.layers:
    print(lyr.input)
print('output')
for lyr in model.layers:
    print(lyr.output)
grad_model = tf.keras.models.Model(inputs=[model.get_layer(model_layer).input, model.input],
                                   outputs=[
                                            model.get_layer(model_layer).get_layer(layer_name).output])
idx = 0
for i in range(10):
    imgs, lbls, files = gen.datagen()
    nplbls = lbls.numpy()
    img = np.reshape(imgs[0], (1, 224, 224, 3))
    lbl = np.reshape(nplbls[0], (1, 2))
    with tf.GradientTape() as tape:
        g = grad_model(img)
        predictions = model(img)
        loss = predictions[:, idx]
        plt.imshow(g)
        plt.show()
