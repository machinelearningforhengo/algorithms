import tensorflow as tf 
import numpy as np 
# from function import create_samples 
from function import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500 
seed = 700
embiggen_factor = 70 

np.random.seed(seed)
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

model = tf.global_variables_initializer()

with tf.Session() as sess: 
    sample_values = sess.run(samples)
    #centroid_values = sess.run(centroids)
    updated_centroid_value = sess.run(updated_centroids)
    print(updated_centroid_value)
#plot_cluster(sample_values, centroid_values, n_samples_per_cluster)
plot_cluster(sample_values, updated_centroid_value, n_samples_per_cluster)