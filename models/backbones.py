"""
Unified model builder: any backbone x any task head.

`build_model()` pairs a convolutional backbone with a task head from
`models.heads`, so the same code trains an alive/dead classifier, a
continuous predictor, or a time-to-event survival model. It also exposes the
name of the last spatial conv layer for each backbone so the XAI engine
(`activationmap.explain`) can target it without the caller guessing.

Backbones
---------
    'vgg16', 'vgg19', 'resnet50'   : original GEDI-ORDER options
    'efficientnetv2s'              : EfficientNetV2-S (stronger, efficient)
    'convnext_tiny'               : ConvNeXt-Tiny (modern conv net, ViT-competitive)

Preprocessing note
------------------
The existing datagenerator emits images normalized to [0, 1] then
VGG-mean-subtracted (BGR). `build_model(include_preprocessing=...)` controls
whether the backbone's native ImageNet preprocessing is prepended. Leave it
False to stay compatible with the current tfrecord pipeline; set it True if
you feed raw 0-255 RGB images.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from models import heads

# Backbone factory + the last spatial conv layer to target for Grad-CAM.
_BACKBONES = {
    "vgg16": (tf.keras.applications.VGG16, "block5_conv3"),
    "vgg19": (tf.keras.applications.VGG19, "block5_conv4"),
    "resnet50": (tf.keras.applications.ResNet50, "conv5_block3_out"),
    "efficientnetv2s": (tf.keras.applications.EfficientNetV2S, "top_conv"),
    "convnext_tiny": (getattr(tf.keras.applications, "ConvNeXtTiny", None),
                      "convnext_tiny_stage_3_block_2_depthwise_conv"),
}

# Loss + eval metric appropriate to each task (survival metrics are computed
# out-of-graph via models.heads.concordance_index).
_TASK_COMPILE = {
    "classification": ("categorical_crossentropy", ["accuracy"]),
    "regression": ("mse", ["mae"]),
    "survival_cox": (heads.cox_partial_likelihood_loss, []),
    "survival_discrete": (None, []),  # loss built from num_bins at compile time
}


def gradcam_layer_for(backbone: str) -> str:
    """Return the default Grad-CAM target layer for a backbone name."""
    if backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}; expected {list(_BACKBONES)}")
    return _BACKBONES[backbone][1]


def build_model(backbone: str = "resnet50",
                task: str = "classification",
                num_outputs: int = 2,
                imsize: Tuple[int, int, int] = (224, 224, 3),
                fine_tune_at: float = 0.5,
                learning_rate: float = 1e-4,
                optimizer: str = "adam",
                weight_decay: float = 1e-5,
                hidden: Tuple[int, ...] = (256,),
                dropout: float = 0.3,
                include_preprocessing: bool = False,
                weights: Optional[str] = "imagenet",
                compile_model: bool = True) -> Model:
    """Build (and optionally compile) a backbone + task head.

    Args:
        backbone: key of `_BACKBONES`.
        task: one of `heads.ALL_TASKS`.
        num_outputs: classes / targets / time bins (see build_head).
        imsize: input shape.
        fine_tune_at: fraction of backbone layers (from the input side) kept
            frozen; the top (1 - fine_tune_at) fraction is trainable. 0 trains
            all, 1 freezes the whole backbone (feature extraction).
        learning_rate, optimizer, weight_decay: optimizer settings.
        hidden, dropout: head configuration.
        include_preprocessing: prepend backbone-native ImageNet preprocessing.
        weights: 'imagenet' or None.
        compile_model: compile with the task-appropriate loss.

    Returns:
        A tf.keras.Model. For survival_discrete, num_outputs is the number of
        time bins and the model is compiled with a discrete-time hazard loss.
    """
    if backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}; expected {list(_BACKBONES)}")
    if task not in heads.ALL_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected {heads.ALL_TASKS}")

    ctor, _ = _BACKBONES[backbone]
    if ctor is None:
        raise RuntimeError(
            f"Backbone {backbone!r} is unavailable in this TensorFlow build "
            f"({tf.__version__}); upgrade TF or pick another backbone.")

    inputs = layers.Input(shape=imsize, name="input_1")
    x = inputs
    if include_preprocessing:
        # Each app module ships its own preprocess_input.
        module = getattr(tf.keras.applications, backbone.split("_")[0], None)
        if module is not None and hasattr(module, "preprocess_input"):
            x = layers.Lambda(module.preprocess_input, name="preprocess")(x)

    base = ctor(include_top=False, weights=weights, input_tensor=x,
                input_shape=imsize)

    # Partial fine-tuning: freeze the bottom `fine_tune_at` fraction.
    n = len(base.layers)
    freeze_until = int(round(fine_tune_at * n))
    for i, layer in enumerate(base.layers):
        # BatchNorm layers stay frozen when their block is frozen to keep
        # ImageNet running stats stable during fine-tuning.
        layer.trainable = i >= freeze_until

    feat = layers.GlobalAveragePooling2D(name="gap")(base.output)
    outputs = heads.build_head(feat, task=task, num_outputs=num_outputs,
                               hidden=hidden, dropout=dropout, name="head")
    model = Model(inputs=base.input, outputs=outputs,
                  name=f"{backbone}_{task}")

    if compile_model:
        opt = _make_optimizer(optimizer, learning_rate, weight_decay)
        loss, metrics = _TASK_COMPILE[task]
        if task == "survival_discrete":
            loss = heads.discrete_time_hazard_loss(num_outputs)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def _make_optimizer(name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "adamw":
        try:
            import tensorflow_addons as tfa
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        except Exception:
            # TF>=2.11 has a native AdamW.
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}")
