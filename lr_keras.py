import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts), np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).T

# X is all of the points
X = np.vstack((Xa, Xb))
# Y is all of the labels. Things in the top region have
# a label of 0 and things in the bottom a label of 1
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

# gradient = (point.T(p-y))*alpha/m
# line_parameters = line_parameters - gradient

# Adam uses Stochastic gradient descent. We don't need
# to look at all points to work out gradient. Just a sample
adam = Adam(learning_rate = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
# X is the data, Y are the labels
h = model.fit(x=X, y=Y, verbose=1, batch_size = 50, epochs=10, shuffle='true')

plt.plot(h.history['loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['Loss'])

# plt.scatter(X[:n_pts, 0], X[:n_pts, 1], color='r')
# plt.scatter(X[n_pts:, 0], X[n_pts:, 1], color='b')
plt.show()


