from math import *
from time import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array

n = a([[0, 0, 1.0]]).T  # plane normal
sensor_origin = a([0, 0, 1.0])  # sensor origin

P = a([0, 0, 0])  # plane origin

N = int(32 * (360 / 0.2))
N = int(10)
FOV = 2 * pi
# R = a([[cos(FOV * i / N), sin(FOV * i / N), -1] for i in range(N)])
R = a([[1, 0, -cos(FOV * i / N)] for i in range(N)])

R = R / np.linalg.norm(R, axis=-1)[:, np.newaxis]  # make unit length

O = np.repeat(sensor_origin[:, np.newaxis], N, axis=1).T  # simple ray origin
O[...,2] += np.linspace(-N/10, 0, N)

start = time()
rays = np.hstack((O, R))

#Cone specifications
V = a([[1, 0, 0]]).T
D = a([[0, 0, -1]]).T
base_diameter = 0.15
h = 0.31
# th = pi/4
th = np.arctan((base_diameter/2)/h)
M = D @ D.T - cos(th) * np.identity(3)

C2 = []
C1 = []
C0 = []
T = []

for (ray, origin) in zip(R, O):
    U = ray[np.newaxis].T
    P = origin[np.newaxis].T
    delta = (P - V)

    print('U', U)
    print('P', P)
    print('Delta', delta)

    c2 = U.T @ M @ U
    c1 = U.T @ M @ delta
    c0 = delta.T @ M @ delta

    C2.append(c2)
    C1.append(c1)
    C0.append(c0)

    assert(c2.shape == (1,1))
    assert(c1.shape == (1,1))
    assert(c0.shape == (1,1))

    ddd2 = c1 ** 2 - c0*c2
    ddd = np.sqrt(ddd2)
    t = (-c1 + ddd) / c2
    print('t', t)

    T.append(t)

C2 = np.hstack(C2)
C1 = np.hstack(C1)
C0 = np.hstack(C0)
T = np.hstack(T)
print('T\n', T)

U = R.T
P = O.T

delta = (P - V)

# c2 = U.T @ M @ U
# c1 = U.T @ M @ delta
# c0 = delta.T @ M @ delta
#Directly compute diagonal of the matrix products
c2 = (U.T.dot(M) * U.T).sum(-1)
c1 = (U.T.dot(M) * delta.T).sum(-1)
c0 = (delta.T.dot(M) * delta.T).sum(-1)

ddds = np.sqrt(c1**2 - c0*c2)

low_soln = (-c1-ddds)/c2
high_soln = (-c1+ddds)/c2
T = np.minimum(low_soln, high_soln)[..., np.newaxis]
print('T\n', T)

#prune solutions that aren't in bounds of cone we are considering
height_condition = delta.T @ D + np.multiply(U.T @ D, T)
satisfies_cond = np.all((height_condition >= 0, height_condition <= h), axis = 0).flatten()
T = T[satisfies_cond]
O = O[satisfies_cond]
R = R[satisfies_cond]
print('T\n', T)

print('C2\n', C2)
print('c2\n', c2)
print('C1\n', C1)
print('c1\n', c1)
print('C0\n', C0)
print('c0\n', c0)


assert(np.isclose(C2, c2).all())
assert(np.isclose(C1, c1).all())
assert(np.isclose(C0, c0).all())

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
