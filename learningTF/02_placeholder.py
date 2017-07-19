import tensorflow as tf 

x = tf.placeholder("float", None)
y = x * 2 

with tf.Session() as sess: 
    x_data = [[1,2,3], [4,5,6]]
    result = sess.run(y, feed_dict={x:x_data})
    print(result)