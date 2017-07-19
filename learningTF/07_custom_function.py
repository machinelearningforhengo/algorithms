import tensorflow as tf 
from matplotlib import pyplot as plt 

shape=(50,50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
with tf.Session() as session: 
    X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys', interpolation='nearest')


import numpy as np 
from scipy.signal import convolve2d 

def update_board(X):
    N = convolve2d(X, np.ones((3,3)), mode='same', bourndary='wrap') - X 
    X = (N == 3) | (X & (N == 2))
    return X 

board = tf.placeholder(tf.int32, shape=shape, name = 'board')
board_update = tf.py_func(update_board, [board], [tf.int32])
with tf.Session() as sess: 
    initial_board_values = sess.run(initial_board)
    X = sess.run(board_update, feed_dict={board: initial_board_values})[0] 

    import matplotlib.animation as animation 
    def game_of_life(*args):
        X = sess.run(board_update, feed_dict={board: X})[0]
        plot.set_array(X)
        return plot 

    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
    plt.show()

