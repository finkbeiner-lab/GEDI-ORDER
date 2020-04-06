import tensorflow as tf
from ops.gradcam_ops import Gradcam
import preprocessing.datagenerator as pipe
import param_gedi as param
import os
import numpy as np

p = param.Param()
layer_name = 'block5_conv3'
# layer_name = 'batch_normalization_3'
tfrecord = p.data_train
DEBUG = True
gen = pipe.Dataspring(tfrecord)
gen.datagen_base(istraining=False)
timestamp = 'vgg16_2020_04_06_15_23_12'
import_path = os.path.join(p.models_dir, "{}.h5".format(timestamp))

print('Loading model...')
model = tf.keras.models.load_model(import_path)
model.trainable = False
model.summary()
# print([layer.output.name for layer in model.get_layer('sequential').layers])
gcam = Gradcam(model, model_layer='sequential', layer_name=layer_name, debug=DEBUG)
for i in range(10):
    # image_batch, lbl_batch = DatTest.datagen()
    # # Plops.show_batch(image_batch, lbl_batch)

    imgs, lbls, files = gen.datagen()
    # imgs = np.array(imgs)
    nplbls = lbls.numpy()
    inputimg = np.reshape(imgs[0], (1, 224, 224, 3))
    inputlbl = np.reshape(nplbls[0], (1, 2))
    gcam.guided_backprop(inputimg, inputlbl)
