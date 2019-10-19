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
N = int(1)
R = a([[cos(2 * pi * i / N), sin(2 * pi * i / N), -1] for i in range(N)])

R = R / np.linalg.norm(R, axis=-1)[:, np.newaxis]  # make unit length

O = np.repeat(sensor_origin[:, np.newaxis], N, axis=1).T  # simple ray origin

start = time()
rays = np.hstack((O, R))

V = a([1, 0, 1])
D = a([[0, 0, -1]]).T
th = pi/2
M = D @ D.T - cos(th) * np.identity(3)
delta = (O - V).T

print(delta)

c2 = R @ M @ R.T
c1 = R @ M @ delta.T
c0 = delta @ M @ delta.T

print(c2.shape)
print(c1.shape)
print(c0.shape)

ddd2 = c1 ** 2 - c0*c2
ddd = np.sqrt(ddd2)
print(ddd2)
# T = (-c1 - )

# Solution for plane
# T = ((P - O) @ n) / (R @ n)
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
