import tensorflow as tf


@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad

with tf.Graph().as_default() as g:
  x = tf.Variable(5.0)
  with tf.GradientTape() as tape:
    s_2 = log1pexp(x)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(tape.gradient(s_2, x)))