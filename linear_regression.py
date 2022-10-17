import numpy as np
import matplotlib.pyplot as plt

n_pts = 10
np.random.seed(0)
bias = np.ones(n_pts)

random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)
top_region = np.array([random_x1_values, random_x2_values, bias]).T

bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T

all_points = np.vstack((top_region, bottom_region))
print(all_points)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
plt.show()
