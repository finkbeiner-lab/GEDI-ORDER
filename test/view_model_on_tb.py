import tensorflow as tf
import param_gedi as param

p = param.Param()

model = tf.keras.models.load_model(p.base_gedi, compile=True)

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)
