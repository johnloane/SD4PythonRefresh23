import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

assert(X_train.shape[0] == y_train.shape[0]), "The number of training images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to the number of labels"
assert(X_train.shape[1:] == (28,28)), "The dimensions of the training images are not all 28x28"
assert(X_test.shape[1:] == (28,28)), "The dimensions of the test images are not all 28x28"

num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train==j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i==2:
            num_of_samples.append(len(x_selected))

plt.figure(figsize=(12,4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# One hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_train = X_train/255
X_test = X_test/255



