import tensorflow as tf 
import numpy as np 

# x = tf.Variable(0, name='x')

# threshold = tf.constant(5)

# model = tf.global_variables_initializer()

# with tf.Session() as sess: 
#     sess.run(model)

#     while sess.run(tf.less(x, threshold)):
#         x += 1
#         print(sess.run(x))


#Gradient Descent
x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable([1.0, 2.0], name="w")
y_ = tf.multiply(x, w[0]) + w[1]

err = tf.square(y - y_)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(err)
model = tf.global_variables_initializer()

errors = []
with tf.Session() as sess:
    sess.run(model)
    for i in range(1000):
        x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        y_train = x_train * 2 + 6
        x_value, y_value = sess.run([train_op, y_train])
        _, error_value = sess.run([train_op, err], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)            
    w_value = sess.run(w)
    print("Predicted model: {a: 0.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

import matplotlib.pyplot as plt
plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")