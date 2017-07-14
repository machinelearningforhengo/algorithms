from tensorflow.contrib.tensorboard.plugins import projector

N = 10000 #number of items
D = 200 # dimensionality of the embedding

embedding_var = tf.Variable(tf.random_normal([N, D], name = 'word_embedding')

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name 

embedding.metadata_path = os.path.join('./tmp/projector_model', 'metadata.tsv')
summary_writer = tf.summary.FileWriter('./tmp/projector_model')

projector.visualize_embeddings(summary_writer, config)