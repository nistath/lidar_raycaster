from typing import NamedTuple, List, Tuple
from lib6003.image import png_write

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class Range(NamedTuple):
    low: float
    high: float
    step: float

    @classmethod
    def from_string(cls, string):
        return cls(*map(float, string.strip().split(',')))


class Grid(NamedTuple):
    ranges: Tuple[Range, Range]
    grid: np.array

    @classmethod
    def from_csv(cls, fname):
        f = open(fname)
        ranges = tuple(Range.from_string(f.readline()) for _ in range(2))
        grid = np.loadtxt(f, delimiter=",").T

        return cls(ranges, grid)

    def plot(self):
        # plt.xlim(self.ranges[0].low, self.ranges[0].high)
        # plt.ylim(self.ranges[1].low, self.ranges[1].high)
        plt.imshow(self.grid, cmap='viridis', origin='lower', interpolation='nearest', extent=[
                   self.ranges[0].low, self.ranges[0].high, self.ranges[1].low, self.ranges[1].high])
        plt.ylabel('Vertical Mounting (m)')
        plt.xlabel('Longitudinal Mounting (m)')
        plt.colorbar()


car = Image.open('/home/nistath/Downloads/car.png')
pix2meter = 752 / 1.530

if True:
    grid1 = Grid.from_csv('/home/nistath/Git/rt/depp.csv')
    grid2 = Grid.from_csv('/home/nistath/Git/rt/earr.csv')

    grid1.plot()
    plt.show()
    grid2.plot()
    plt.show()

    grid = Grid(grid1.ranges, (grid1.grid + grid2.grid)/2)
else:
    grid = Grid.from_csv('/home/nistath/Git/rt/earr.csv')

png_write(grid.grid, '/home/nistath/Desktop/grid.png', 'topleft', True)
grid.plot()
# plt.imshow(car, alpha=0.1)
plt.show()
