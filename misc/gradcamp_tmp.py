import tensorflow as tf

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    dtype = op.inputs[0].dtype
    gate_f = tf.cast(op.outputs[0] > 0, dtype) #for f^l > 0
    gate_R = tf.cast(grad > 0, dtype) #for R^l+1 > 0
    return gate_f * gate_R * grad

with tf.Graph().as_default().gradient_override_map({'Relu': 'GuidedRelu'}):
    with tf.GradientTape() as tape:
        x = tf.constant([10., 2.])
        tape.watch(x)
        y = tf.nn.relu(x)
        z = tf.reduce_sum(-y ** 2)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            print(sess.run(x))
            print(sess.run(y))
            print(sess.run(z))
            print(sess.run(tape.gradient(z,x)))
            # print(x.numpy())
            # print(y.numpy())
            # print(z.numpy())
            # print(tape.gradient(z, x).numpy())