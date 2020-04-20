import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

IMAGE_PATH = '/home/jlamstein/PycharmProjects/GEDI-ORDER/examples/cat.3.jpg'
LAYER_NAME = 'block5_conv3'
model_layer = 'vgg16'
CAT_CLASS_INDEX = 281

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
plt.figure()
plt.imshow(img)
img = tf.keras.preprocessing.image.img_to_array(img)
imsize = (224,224,3)

input = layers.Input(shape=(imsize[0], imsize[1], imsize[2]))
base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input,
                                          input_shape=(imsize[0], imsize[1], imsize[2]))
# base_model.trainable = False
flat = layers.Flatten()
dropped = layers.Dropout(0.5)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

fc1 = layers.Dense(16, activation='relu', name='dense_1')
fc2 = layers.Dense(16, activation='relu', name='dense_2')
fc3 = layers.Dense(128, activation='relu', name='dense_3')
prediction = layers.Dense(2, activation='softmax', name='output')
for layr in base_model.layers:
    if ('block5' in layr.name):

        layr.trainable = True
    else:
        layr.trainable = False

# x = base_model(input)
block5_pool = base_model.get_layer('block5_pool')
x = global_average_layer(block5_pool.output)
x = fc1(x)
x = fc2(x)
x = prediction(x)

model = tf.keras.models.Model(inputs = input, outputs = x)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
# Create a graph that outputs target convolution and output
# grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])
grad_model = tf.keras.models.Model(inputs = [model.input],
                                   outputs=[model.output,
                                            model.get_layer(model_layer).output])
print(model.get_layer(model_layer).get_layer(LAYER_NAME).output)
# Get the score for target class

# Get the score for target class
with tf.GradientTape() as tape:
    predictions, conv_outputs = grad_model(np.array([img]))
    loss = predictions[:, 1]

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

# Heatmap vis
cam = cv2.resize(cam.numpy(), (224, 224))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

plt.figure()
plt.imshow(output_image)
plt.show()