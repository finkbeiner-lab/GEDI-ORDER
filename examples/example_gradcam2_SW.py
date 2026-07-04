import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pdb

IMAGE_PATH = '/gladstone/finkbeiner/linsley/Shijie_ML/Tau_PFF/Mito/PFF_LIPO_T8-12/PID20220313_2022-0309-MsNeuron-219-Tau-coTrans-PFFlipo_T8_96.0-0_E7_0_Epi-RFP16_0_0_1_BGs_MN_ALIGNED_12.tif'
LAYER_NAME = 'block5_conv4'
#model_layer = 'vgg19'
CAT_CLASS_INDEX = 1


# edited by SHIJIE

from tensorflow import keras
model = keras.models.load_model('/gladstone/finkbeiner/linsley/Shijie_ML/Tau_PFF/Mito/CNN/saved_models/vgg19_2022_11_02_23_49_25.h5')


img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
plt.figure()
plt.imshow(img)

img = tf.keras.preprocessing.image.img_to_array(img)

# Load initial model
#model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(LAYER_NAME).output, model.output])
print(model.get_layer(LAYER_NAME).output)
# Get the score for target class
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img]))
    loss = predictions[:, CAT_CLASS_INDEX]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

# Apply guided backpropagation
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = gate_f * gate_r * grads

# Average gradients spatially
weights = tf.reduce_mean(guided_grads, axis=(0, 1))

# Build a ponderated map of filters according to gradients importance
cam = np.ones(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, index]

# Heatmap vis_old
cam = cv2.resize(cam.numpy(), (224, 224))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
plt.imshow(cam)
plt.show()


output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

plt.figure()
plt.imshow(output_image)
plt.show()
