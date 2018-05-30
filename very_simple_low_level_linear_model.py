import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)

# tf.constant(5, name="input_a")
# tf.constant(5, name="input_b")

session = tf.Session()
print(session.run([node1, node2]))

node3 = node1 + node2
print("node 3", node3)
print("session run node3", session.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print("adder node scalar:", session.run(adder_node, {a:3, b:4.5}))
print("adder node array:", session.run(adder_node, {a:[1,3], b:[2,4]}))

adder_and_triple_node = adder_node * 3
print("adder and triple node:", session.run(adder_and_triple_node, {a:3, b:4.5}))
