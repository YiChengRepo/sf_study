import numpy as np
import tensorflow as tf

### Model parameters ###
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

### Model input and output ###
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
init = tf.global_variables_initializer()
# ### training data ###
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

sess = tf.Session()
sess.run(init)
# fixed value to get 0 loss , just as a example !
# fixW = tf.assign(W, [-1.])
# fixB = tf.assign(b, [1.])
# sess.run([fixW, fixB])
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# print(sess.run(loss, {x: x_train, y: y_train}))
# print out 0 !

### loss ###
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# ### optimizer ###
# ### training loop ###
sess.run(init)  # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

print('final result: ')
print(sess.run([W, b]))

