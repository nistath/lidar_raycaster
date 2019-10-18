from math import *
from time import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array

n = a([[0, 0, 1]]).T  # plane normal
sensor_origin = a([0, 0, 1])  # sensor origin

P = a([0, 0, 0])  # plane origin

N = int(32 * (360 / 0.2))
N = int(20)
R = a([[cos(2 * pi * i / N), sin(2 * pi * i / N), -1] for i in range(N)])

R = R / np.linalg.norm(R, axis=-1)[:, np.newaxis]  # make unit length

print(R.shape)
O = np.repeat(sensor_origin[:, np.newaxis], N, axis=1).T  # simple ray origin

start = time()
rays = np.hstack((O, R))

T = ((P - O) @ n) / (R @ n)
points = O + R * T
print(time() - start)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

ax.quiver(*rays.T)
ax.scatter(*points.T)

plt.show()
