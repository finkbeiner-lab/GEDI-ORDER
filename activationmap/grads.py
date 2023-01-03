from contextlib import contextmanager

from cv2 import resize
import numpy as np
import tensorflow as tf
from tensorflow import RegisterGradient
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
# from memory_profiler import profile
import sys
from pympler import asizeof
from preprocessing.datagenerator import Dataspring


def mem(obj, name):
    m = asizeof.asizeof(obj)
    print(name, m)


@RegisterGradient('GuidedRelu')
def guided_relu(op, grad):
    grad_filt = tf.cast(grad > 0, 'float32')
    out_filt = tf.cast(op.outputs[0] > 0, 'float32')

    return out_filt * grad_filt * grad


class Grads:
    def __init__(self, model_path, guidedbool, summary=True):
        # change this
        """
        Constructor takes path for h5 model, generating internal guided and unguided instances
        :param model_path: filename
        :param summary: whether to display model summaries
        """

        self.normal = self.new_context_pair()
        self.guided = self.new_context_pair()
        self.guidedbool = guidedbool
        self.raw_func = None
        self.layer_func = None
        self.verbose = False

        with Grads.using_context(*self.normal):
            self.model_n = load_model(model_path)
        with Grads.using_context(*self.guided, {'Relu': 'GuidedRelu'}):
            self.model_g = load_model(model_path)

        if summary:
            self.model_n.summary()
            self.model_g.summary()

    @staticmethod
    def new_context_pair():
        """

        Returns: tuple of tf.Graph object, linked tf.Session object for managing context of multiple models

        """

        g = tf.compat.v1.Graph()
        s = tf.compat.v1.Session(graph=g)

        return g, s

    @staticmethod
    @contextmanager
    def using_context(graph, sess, override_map=None):
        """
        Method provides a context (for use in 'with' statement) where model associated with graph, session pair can be safely used

        Args:
            graph: tf.Graph object
            sess: tf.Session object linked to graph
            override_map: dictionary mapping default operation names to names of registered gradients

        """

        if override_map is None:
            with graph.as_default(), sess.as_default():
                yield
        else:
            # note: override_map only need be applied once to new graph (i.e. during execution of load_model)
            with graph.as_default(), sess.as_default(), graph.gradient_override_map(override_map):
                yield

    @contextmanager
    def using_default(self, guided=False):
        """
        Method provides context using default models (either normal or guided)

        Args:
            guided: if True use guided model-associated context

        """

        pair = self.guided if guided else self.normal
        with Grads.using_context(*pair):
            yield

    @staticmethod
    #    @profile
    def raw_grad_func(model, pick_preds=True, ret_preds=False):
        """
        Returns a context-dependent function computing gradients of output class(es) with respect to inputs

        Args:
            model: keras model object
            pick_preds: if True pick classes to compute gradients for according to model's top prediction
            ret_preds: if True returns predictions of model for each input

        Returns: a Keras function

        """
        print('raw grad func')
        inp = model.input
        preds = model.output
        out = preds.op.inputs[0]  # eliminates activation layer interference

        inp_arr = [inp]
        out_arr = []
        if pick_preds:
            logits = K.max(out, axis=1)
            out_arr.append(K.gradients(logits, inp)[0])
        else:
            class_var = K.placeholder(ndim=0, dtype='int32')
            inp_arr.append(class_var)
            logits = out[:, class_var]
            out_arr.append(K.gradients(logits, inp)[0])

        if ret_preds:
            out_arr.append(preds)

        return K.function(inp_arr, out_arr)

    @staticmethod
    #    @profile
    def layer_grad_func(model, layer_name, pick_preds=True, ret_preds=False):
        """
        Returns a context-dependent function computing gradients of output class(es) with respect layer values

        Args:
            model: keras model object
            layer_name: name of layer to compute gradients with respect to
            pick_preds: if True pick classes to compute gradients for according to model's top prediction
            ret_preds: if True returns predictions of model for each input

        Returns: a Keras function

        """
        print('layer_grad_func')
        inp = model.input
        preds = model.output
        out = preds.op.inputs[0]
        model.summary()
        layer = model.get_layer(layer_name).output

        inp_arr = [inp]
        out_arr = []
        if pick_preds:
            logits = K.max(out, axis=1)
            out_arr.append(K.gradients(logits, layer)[0])
        else:
            class_var = K.placeholder(ndim=0, dtype='int32')
            inp_arr.append(class_var)
            logits = out[:, class_var]
            out_arr.append(K.gradients(logits, layer)[0])
        out_arr.append(layer)

        # mem(out_arr, 'out_arr')
        # mem(inp_arr, 'inp_arr')
        if ret_preds:
            out_arr.append(preds)

        return K.function(inp_arr, out_arr)

    #    @profile
    def batch_grads(self, batch, guided=False, class_id=None, ret_preds=False):
        """
        Operates on a batch of inputs; returns class gradient wrt. respective inputs

        Args:
            batch: tensor of inputs (at least 2d; 1st dimension batch dimension)
            guided: whether to use self class instance's guided backprop or unmodified model configuration
            class_id: if None, use gradient of predicted class; otherwise all gradients of the specified class (integer index)
            ret_preds: whether to return predictions

        Returns: A tensor of same shape as inputs with unnormalized gradients of the specified class(es)

        Ideas for usage:
            Iterate once over entire batch with class_id=None,
            Compare predicted class with actual label,
            Batch remaining (incorrectly predicted images) according to label,
            Iterate by label over all incorrectly predicted images of that label,
            Thus obtaining gradients for each image both of predicted and correct class;

            The above may apply to all batch methods that follow

        """
        print('batch grads')
        with self.using_default(guided=guided):
            if self.raw_func is None:
                self.raw_func = Grads.raw_grad_func(self.model_g if guided else self.model_n,
                                                    pick_preds=class_id is None,
                                                    ret_preds=True)
            # func = Grads.raw_grad_func(self.model_g if guided else self.model_n, pick_preds=class_id is None,
            #                            ret_preds=True)
            if class_id is None:
                grads, preds = self.raw_func([batch])
            else:
                grads, preds = self.raw_func([batch, class_id])
        if self.verbose:
            mem(grads, 'grads')
            mem(preds, 'preds')
            mem(self.model_g, 'model_g')
            mem(self.model_n, 'model_n')
        if ret_preds:
            return [grads, preds]
        return [grads]

    #    @profile
    def batch_heatmaps(self, batch, layer_name, class_id=None, ret_preds=False):
        print('batch_heatmaps')
        with self.using_default():
            if self.layer_func is None:
                self.layer_func = Grads.layer_grad_func(self.model_n, layer_name, pick_preds=class_id is None,
                                                        ret_preds=True)
            if class_id is None:
                grads, layer, preds = self.layer_func([batch])
            else:
                grads, layer, preds = self.layer_func([batch, class_id])

        heatmaps = []
        print('batch_heatmaps_loop')
        for i in range(layer.shape[0]):  # batchwise loop
            for j in range(layer.shape[-1]):  # channelwise loop
                grad_ij = np.sum(grads[i, ..., j])
                layer[i, ..., j] *= grad_ij
                # layer[i, ..., j] *= np.sum(grads[i, ..., j])
            heatmaps.append(np.sum(layer[i], axis=-1))
        heatmaps = np.maximum(np.array(heatmaps), 0)  # ReLU
        if self.verbose:
            mem(batch, 'batch')
            mem(heatmaps, 'heatmaps')
            mem(layer, 'layer')
        print('return_batch_heatmaps')
        if ret_preds:
            return [heatmaps, preds]
        return [heatmaps]

    #    @profile
    def guided_gradcam(self, batch, layer_name, class_id=None, ret_preds=False):
        grads, preds = self.batch_grads(batch, guided=self.guidedbool, class_id=class_id, ret_preds=True)
        heatmaps, _ = self.batch_heatmaps(batch, layer_name, class_id=class_id, ret_preds=True)

        for i in range(grads.shape[0]):
            grads[i] = np.maximum(grads[i], 0)  # ReLU/positively contributing pixel slice
            heatmap = resize(heatmaps[i], (224, 224))  # interpolate heatmap over entire pixel area
            # ^^ may wish to use custom interpolation instead

            for ch in range(grads[i].shape[-1]):
                grads[i, ..., ch] *= heatmap

            if np.amax(grads[i]) > 0: grads[i] /= np.amax(grads[i])  # 0-1 normalization over composite image
        if self.verbose:
            mem(grads, 'guided grads')
            mem(heatmaps, 'guided heatmaps')
        if ret_preds:
            return [grads, preds]
        return [grads]

    def process_images(self, imgs, lbls):
        Dat = Dataspring(None)
        dataset = tf.data.Dataset.from_tensor_slices((imgs, lbls))
        ds = Dat.datagen_base(istraining=False, count=1, dataset=dataset)
        return ds




    def guided_gradcam_gray(self, batch, layer_name, class_id=None, ret_preds=False):
        """
        Guided gradcam
        Args:
            batch: imgs
            layer_name: name of model layer to image
            class_id: labels
            ret_preds:

        Returns:

        """
        print('guided gradcam gray')
        # guided backprop
        grads, preds = self.batch_grads(batch, guided=self.guidedbool, class_id=class_id, ret_preds=True)
        # gradCAM heatmaps from targeted layer_name
        heatmaps, _ = self.batch_heatmaps(batch, layer_name, class_id=class_id, ret_preds=True)

        grads_final = []
        for i in range(grads.shape[0]):
            # consider other grayscale methods
            grads[i] = np.maximum(grads[i], 0)
            grad = np.max(grads[i], axis=-1)
            heatmap = resize(heatmaps[i], (224, 224))
            # _f = '/mnt/finkbeinerlab/robodata/Josh/Gradcam/results/batches16bit/10_single_unnormalized'
            # np.save(_f + f'/neg_backprop_grads_label_{class_id}.npy', grad)
            # np.save(_f + f'/neg_gradcam_label_{class_id}.npy', heatmap)
            if self.guidedbool:
                # multiply guided backprop * gradCAM heatmap
                grad *= heatmap
                # np.save(_f + f'/neg_guided_label_{class_id}.npy', grad)

            if np.amax(grad) > 0: grad /= np.amax(grad)
            grads_final.append(grad)
        if self.verbose:
            mem(grads, 'gray guided grads')
            mem(heatmaps, 'gray guided heatmaps')
            mem(grad, 'gray guided grad')
            mem(heatmap, 'gray guided heatmap')
            mem(batch, 'gray batch')
        if ret_preds:
            return [grads_final, preds]
        return [grads_final]

    #    @profile
    def gen_ggcam_stacks(self, imgs, lbls, layer_name, ret_preds=True, gray_morphology=False):
        """
        Gets gradcam for both labels.
        Args:
            imgs:
            lbls:
            layer_name:
            ret_preds:

        Returns:

        """
        # defaults to 8bit
        print('gen ggcam stacks')

        def tif_format(img):
            img = img.astype(dtype=np.float)
            img -= np.amin(img)
            img /= np.amax(img)
            img *= 255.
            img = img.astype(dtype=np.uint8)
            return img

        grads_0, preds = self.guided_gradcam_gray(imgs, layer_name, class_id=0, ret_preds=True)
        grads_1, preds = self.guided_gradcam_gray(imgs, layer_name, class_id=1, ret_preds=True)

        for img, lbl, grad_0, grad_1, pred in zip(imgs, lbls, grads_0, grads_1, preds):
            grad_pair = [grad_0, grad_1]
            # if np.argmax(lbl) == 1:
            #     grad_pair = grad_pair[::-1]
            gray = np.sum(img, axis=-1)
            gray = gray / np.max(gray)
            show_group = [gray] + grad_pair
            # show_group = [gray,green,blue]
            show_group = list(map(tif_format, show_group))
            gradimg= np.float32(np.dstack(show_group))
            if gray_morphology:
                gradimg[...,0] *= 2
                gradimg[...,1] += gradimg[...,0]
                gradimg[...,2] += gradimg[...,0]
                gradimg /= np.max(gradimg)
                gradimg *= 255
            else:
                gradimg /= np.max(gradimg)

            gradimg *= 255
            gradimg = gradimg.astype(dtype=np.uint8)
            res = [gradimg]
            if self.verbose:
                mem(res, 'res')
                mem(show_group, 'show_group')
            if ret_preds:
                res.append(pred)
            yield res
