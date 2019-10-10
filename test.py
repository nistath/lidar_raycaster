import numpy as np
from math import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array

n = a([[0, 0, 1]]).T  # plane normal
sensor_origin = a([0, 0, 1])  # sensor origin

P = a([0, 0, 0])  # plane origin

N = 10
R = a([[cos(2 * pi * i / N), sin(2 * pi * i / N), -1] for i in range(N)])

R = R / np.linalg.norm(R, axis=-1)[:, np.newaxis]  # make unit length
O = np.repeat(sensor_origin[:, np.newaxis], 10, axis=1).T  # simple ray origin

rays = np.hstack((O, R))

T = ((P - O) @ n) / (R @ n)

points = O + R * T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

ax.quiver(*rays.T)
ax.scatter(*points.T)

plt.show()
