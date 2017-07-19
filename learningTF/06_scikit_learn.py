# scikit-learn is a Python package for data mining & analysis

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt 
import tensorflow as tf

digits = load_digits()

# fig = plt.figure(figsize=(3,3))
# plt.imshow(digits['images'][66], cmap="gray", interpolation=None)
#plt.show()
from sklearn import svm 
classifier = svm.SVC(gamma=0.001)
classifier.fit(digits.data, digits.target)
predicted = classifier.predict(digits.data)

import numpy as np 
print(np.mean(digits.target == predicted))

from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

def input_functions():
    data = tf.constant(X_train)
    target = tf.constant(y_train)
    return data, target

def input_x_test():
    return X_test



from tensorflow.contrib import learn 

n_classes = len(set(y_train))
print(n_classes)
classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("",
                            dimension=X_train.shape[1])], n_classes=n_classes)
classifier.fit(input_fn = input_functions, steps =10)

y_pred = classifier.predict_classes(input_fn = input_x_test)

def input_predict():
    return y_test, y_pred

from sklearn import metrics 
print(metrics.classification_report(y_true=y_test, y_pred = y_pred))