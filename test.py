from math import *
from time import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array

n = a([[0, 0, 1.0]]).T  # plane normal
sensor_origin = a([0, 0, 1.0])  # sensor origin

P = a([0, 0, 0])  # plane origin

HFOV = pi / 8
HBIAS = -HFOV / 2
VFOV = pi / 6
VBIAS = -pi/2

N = 20
rings = 20

R = []
for ring in range(rings):
    z = -2 * cos(VFOV * ring / rings + VBIAS) - 0.5
    R += [[cos(HFOV * i / N + HBIAS),
           sin(HFOV * i / N + HBIAS),
           z] for i in range(N)]

R = a(R)
N = len(R)

R = R / np.linalg.norm(R, axis=-1)[:, np.newaxis]  # make unit length

O = np.repeat(sensor_origin[:, np.newaxis], N, axis=1).T  # simple ray origin
# O[...,2] += np.linspace(-N/10, 0, N)

start = time()

#Cone specifications
h = 0.29
V = a([[1, 0, h]]).T
D = a([[0, 0, -1]]).T
base_diameter = 0.15
# th = pi/4
# th = np.arctan((base_diameter/2)/h)
# M = D @ D.T - cos(th) * np.identity(3)
costh = h / sqrt((base_diameter / 2) ** 2 + h ** 2)
M = D @ D.T - costh * np.identity(3)

C2 = []
C1 = []
C0 = []
T = []

for (ray, origin) in zip(R, O):
    U = ray[np.newaxis].T
    P = origin[np.newaxis].T
    delta = (P - V)

    # print('U', U)
    # print('P', P)
    # print('Delta', delta)

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
    t = min(((-c1 - ddd) / c2, (-c1 + ddd) / c2))
    # print('t', t)

    T.append(t)

C2 = np.hstack(C2)
C1 = np.hstack(C1)
C0 = np.hstack(C0)
T = np.hstack(T).T
print('Told\n', T)

U = R.T
P = O.T

delta = (P - V)

if False:
    ac2 = U.T @ M @ U
    ac1 = U.T @ M @ delta
    ac0 = delta.T @ M @ delta

    assert(np.isclose(np.diagonal(ac2), C2).all())
    assert(np.isclose(np.diagonal(ac1), C1).all())
    assert(np.isclose(np.diagonal(ac0), C0).all())

#Directly compute diagonal of the matrix products
c2 = (U.T @ M * U.T).sum(-1)
c1 = (U.T @ M * delta.T).sum(-1)
c0 = (delta.T @ M * delta.T).sum(-1)
assert(np.isclose(c2, C2).all())
assert(np.isclose(c1, C1).all())
assert(np.isclose(c0, C0).all())

ddds = np.sqrt(c1**2 - c0*c2)

low_soln = (-c1-ddds)/c2
high_soln = (-c1+ddds)/c2
# T = np.minimum(low_soln, high_soln)[..., np.newaxis]
print('T\n', T)

if True:
    #prune solutions that aren't in bounds of cone we are considering
    height_condition = delta.T @ D + np.multiply(U.T @ D, T)
    satisfies_cond = np.all((height_condition >= 0, height_condition <= h), axis = 0).flatten()
    T = T[satisfies_cond]
    O = O[satisfies_cond]
    R = R[satisfies_cond]
print('T\n', T)
rays = np.hstack((O, R))

# Solution for plane
# T = ((P - O) @ n) / (R @ n)
points = O + R * T
print(time() - start)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

ax.quiver(*rays.T, pivot='tail')
ax.scatter(*points.T, color='red')

plt.show()
