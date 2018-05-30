import tensorflow as tf
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="multiply_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")
session = tf.Session()
print(session.run(e))
writer = tf.summary.FileWriter('./my_graph', session.graph)
writer.close()
session.close()

#how to run
#tensorboard --logdir=my_graph/
#TensorBoard 1.8.0 at http://b-0235.local:6006 (Press CTRL+C to quit)

