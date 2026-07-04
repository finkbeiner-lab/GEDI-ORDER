"""
Modern explainable-AI engine for GEDI-ORDER.

A single, eager `tf.GradientTape` implementation that replaces the legacy
TF1 graph/session Guided-Grad-CAM code (`activationmap/grads.py`). It:

    * loads the model once (not twice),
    * works for any number of classes AND for scalar heads (regression,
      Cox survival risk, or a collapsed discrete-hazard risk),
    * offers four complementary attribution methods so a finding that shows
      up under several of them can be trusted:
          - Grad-CAM              (fast, coarse localization)
          - Grad-CAM++            (better for multiple/*small* objects)
          - Score-CAM             (gradient-free, less noisy)
          - Integrated Gradients  (pixel-level, axiomatic attribution)

Usage
-----
    from activationmap.explain import Explainer
    exp = Explainer(model, layer_name="conv5_block3_out")   # or auto-detect
    cam = exp.gradcam(images, target=1)          # class 1
    ig  = exp.integrated_gradients(images)       # scalar/argmax target
    rgb = exp.overlay(raw_image, cam[0])         # uint8 heatmap overlay

All heatmaps are returned as float arrays in [0, 1] with shape (batch, H, W).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last layer that outputs a 4-D (spatial) tensor."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No 4-D convolutional layer found for Grad-CAM.")


class Explainer:
    """Attribution methods over a trained Keras model."""

    def __init__(self, model: tf.keras.Model,
                 layer_name: Optional[str] = None):
        """
        Args:
            model: a trained tf.keras.Model.
            layer_name: conv layer to localize on. If None, the last spatial
                layer is auto-detected.
        """
        self.model = model
        self.layer_name = layer_name or find_last_conv_layer(model)
        conv_layer = model.get_layer(self.layer_name)
        # Sub-model exposing both the conv feature maps and the final output.
        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select(preds: tf.Tensor, target: Optional[Union[int, Sequence[int]]]) -> tf.Tensor:
        """Reduce model output to one scalar per sample to differentiate.

        - scalar heads (regression / survival_cox): use the single output.
        - classification: use `target` class (int, or per-sample list), or the
          argmax class when target is None.
        """
        preds = tf.convert_to_tensor(preds)
        if preds.shape[-1] == 1:
            return preds[:, 0]
        n = tf.shape(preds)[0]
        if target is None:
            idx = tf.argmax(preds, axis=1, output_type=tf.int32)
        elif np.isscalar(target):
            idx = tf.fill([n], int(target))
        else:
            idx = tf.cast(tf.convert_to_tensor(target), tf.int32)
        rows = tf.range(n)
        return tf.gather_nd(preds, tf.stack([rows, idx], axis=1))

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        """Per-sample min-max to [0, 1], ReLU applied by caller if needed."""
        cam = cam.astype(np.float32)
        out = np.empty_like(cam)
        for i in range(cam.shape[0]):
            c = cam[i]
            c = c - c.min()
            m = c.max()
            out[i] = c / m if m > 0 else c
        return out

    def _resize(self, cam: np.ndarray, size) -> np.ndarray:
        """Bilinearly resize (batch, h, w) heatmaps to the model input size."""
        cam4 = cam[..., None]
        resized = tf.image.resize(cam4, size, method="bilinear").numpy()
        return resized[..., 0]

    def _input_hw(self, images) -> tuple:
        return int(images.shape[1]), int(images.shape[2])

    # ------------------------------------------------------------------ #
    # Grad-CAM
    # ------------------------------------------------------------------ #
    def gradcam(self, images, target=None) -> np.ndarray:
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(images, training=False)
            score = self._select(preds, target)
        grads = tape.gradient(score, conv_out)                 # (B, h, w, C)
        weights = tf.reduce_mean(grads, axis=(1, 2))            # (B, C) GAP
        cam = tf.einsum("bhwc,bc->bhw", conv_out, weights)      # weighted sum
        cam = tf.nn.relu(cam).numpy()
        cam = self._resize(cam, self._input_hw(images))
        return self._normalize(cam)

    # ------------------------------------------------------------------ #
    # Grad-CAM++
    # ------------------------------------------------------------------ #
    def gradcam_plus_plus(self, images, target=None) -> np.ndarray:
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(images, training=False)
            score = self._select(preds, target)
            exp_score = tf.exp(score)
        grads = tape.gradient(exp_score, conv_out)              # (B, h, w, C)
        grads = tf.maximum(grads, 0.0)
        g2 = grads ** 2
        g3 = grads ** 3
        sum_conv = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)  # (B,1,1,C)
        denom = 2.0 * g2 + sum_conv * g3
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
        alphas = g2 / denom                                     # (B, h, w, C)
        weights = tf.reduce_sum(alphas * grads, axis=(1, 2))    # (B, C)
        cam = tf.einsum("bhwc,bc->bhw", conv_out, weights)
        cam = tf.nn.relu(cam).numpy()
        cam = self._resize(cam, self._input_hw(images))
        return self._normalize(cam)

    # ------------------------------------------------------------------ #
    # Score-CAM (gradient-free)
    # ------------------------------------------------------------------ #
    def score_cam(self, images, target=None, max_maps: Optional[int] = 32) -> np.ndarray:
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        conv_out, _ = self.grad_model(images, training=False)   # (B, h, w, C)
        conv_out = conv_out.numpy()
        b, h, w, c = conv_out.shape
        in_hw = self._input_hw(images)
        cams = np.zeros((b, in_hw[0], in_hw[1]), dtype=np.float32)

        for i in range(b):
            maps = conv_out[i]                                  # (h, w, C)
            # Optionally keep only the highest-energy channels for speed.
            channels = range(c)
            if max_maps is not None and c > max_maps:
                energy = maps.reshape(-1, c).sum(axis=0)
                channels = np.argsort(energy)[-max_maps:]
            act = np.stack([maps[..., k] for k in channels], axis=0)  # (K, h, w)
            act = tf.image.resize(act[..., None], in_hw, method="bilinear").numpy()[..., 0]
            # Normalize each activation map to [0, 1] to use as an input mask.
            mins = act.reshape(act.shape[0], -1).min(axis=1)[:, None, None]
            maxs = act.reshape(act.shape[0], -1).max(axis=1)[:, None, None]
            act_norm = (act - mins) / (maxs - mins + 1e-8)
            masked = images[i].numpy()[None] * act_norm[..., None]   # (K, H, W, ch)
            preds = self.model.predict(masked, verbose=0)
            weights = self._select(preds, target).numpy()           # (K,)
            weights = np.maximum(weights, 0)
            cam = np.tensordot(weights, act, axes=(0, 0))           # (H, W)
            cams[i] = cam
        return self._normalize(cams)

    # ------------------------------------------------------------------ #
    # Integrated Gradients (pixel-level)
    # ------------------------------------------------------------------ #
    def integrated_gradients(self, images, target=None, baseline=None,
                             steps: int = 50) -> np.ndarray:
        """Axiomatic pixel attribution; returns (batch, H, W) in [0, 1]."""
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        if baseline is None:
            baseline = tf.zeros_like(images)
        else:
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        alphas = tf.linspace(0.0, 1.0, steps + 1)
        total = tf.zeros_like(images)
        for a in alphas:
            interp = baseline + a * (images - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                preds = self.model(interp, training=False)
                score = self._select(preds, target)
            total += tape.gradient(score, interp)
        avg_grads = total / tf.cast(steps + 1, tf.float32)
        ig = (images - baseline) * avg_grads                    # (B, H, W, ch)
        attr = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()       # (B, H, W)
        return self._normalize(attr)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
                colormap: str = "jet") -> np.ndarray:
        """Overlay a [0,1] heatmap on a grayscale/RGB image -> uint8 RGB."""
        import matplotlib.cm as cm

        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

        cmap = cm.get_cmap(colormap)
        colored = cmap(np.clip(heatmap, 0, 1))[..., :3]
        blended = (1 - alpha) * img + alpha * colored
        return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
