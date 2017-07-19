import tensorflow as tf 

# x = tf.constant(35, name = 'x')
# y = tf.Variable(x+5, name = 'y')
#model = tf.global_variables_initializer()
# with tf.Session() as sess:
    # sess.run(model)
    # print(sess.run(y))

# Problem: caculate: y = 5*x^2 - 3*x + 15 with x in (-- 10000)
import numpy as np 

x = np.random.randint(100, size = 1000)

y = tf.Variable(5*x*x - 3*x + 15, name='x')
model = tf.global_variables_initializer()
with tf.Session() as sess:        
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tmp/basic", sess.graph)
    sess.run(model)
    print(x)
    print(sess.run(y))
    
