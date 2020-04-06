"""
Example of gradcam with guided backprop.
https://morioh.com/p/64064daff26c
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import param_gedi as param


# IMAGE_PATH = '/home/jlamstein/PycharmProjects/GEDI-ORDER/examples/cat.3.jpg'
# LAYER_NAME = 'block5_conv3'
# CAT_CLASS_INDEX = 281

class Gradcam:
    def __init__(self, model, model_layer, layer_name, debug):
        self.model = model
        self.p = param.Param()
        # grad_model = tf.keras.models.Sequential()
        # grad_model.add(model.get_layer[index=0])
        # for i in range(100):
        #     layer = model.get_layer(model_layer).get_layer(index=i)
        #     grad_model.add(layer)
        #     if layer.name == layer_name:
        #         break

        # self.grad_model = tf.keras.models.Model(inputs = [model.get_layer(model_layer).input, model.input], outputs=[ model.get_layer(model_layer).get_layer(layer_name).output])
        self.grad_model = tf.keras.models.Model(inputs=[model.input],
                                                outputs=[model.output, model.get_layer(layer_name).output])
        # self.grad_model = grad_model
        self.DEBUG = debug
        self.grad_model.summary()

    def get_grads(self, img, lbl):
        with tf.GradientTape(persistent=True) as tape:
            predictions, conv_outputs = self.grad_model(img)
            # predictions = self.model(img)
            neg_loss = predictions[:, 0]
            pos_loss = predictions[:, 1]

        # Extract filters and gradients
        output = conv_outputs[0]
        neg_grads = tape.gradient(neg_loss, conv_outputs)[0]
        pos_grads = tape.gradient(pos_loss, conv_outputs)[0]
        del tape
        return output, neg_grads, pos_grads

    def get_single_heatmap(self, output, grads):
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

        # Heatmap visualization
        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        return heatmap

    def guided_backprop(self, _img, lbl):
        img = _img.copy()
        output, neg_grads, pos_grads = self.get_grads(img, lbl)
        pos_heatmap = self.get_single_heatmap(output, pos_grads)
        neg_heatmap = self.get_single_heatmap(output, neg_grads)

        # if np.amax(cam) > 0: cam /= np.amax(heatmap)
        # cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        img[:, :, 0] += self.p.VGG_MEAN[0]
        img[:, :, 1] += self.p.VGG_MEAN[1]
        img[:, :, 2] += self.p.VGG_MEAN[2]
        img = np.reshape(img, (224, 224, 3))

        img = np.uint8(img)
        pos_heatmap *= 255
        neg_heatmap *= 255
        pos_heatmap = np.uint8(pos_heatmap)
        neg_heatmap = np.uint8(neg_heatmap)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = np.dstack((gray, pos_heatmap, neg_heatmap))
        # output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

        if self.DEBUG:
            plt.figure()
            plt.imshow(heatmap)
            plt.title('gradcam')
            # plt.figure()
            # # print(np.max(img))
            # # print('min', np.min(img))
            # img = np.abs(img)
            # img = np.uint8(img * 3)
            # plt.imshow(img)

            # plt.title('img')
            plt.show()
