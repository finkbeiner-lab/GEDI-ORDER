import tensorflow as tf
import param_gedi as param
p = param.Param()
base_model = tf.keras.models.load_model(p.base_gedi, compile=True)
for lyr in base_model.layers:
    print(lyr.name)
conv3 = base_model.get_layer('predictions')
tf.print(conv3.get_weights()[1])