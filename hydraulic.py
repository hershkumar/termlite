import matplotlib
from matplotlib.colors import LightSource, LinearSegmentedColormap
import itertools
import noise
import numpy as np
from PIL import Image
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from random import randint
from math import hypot


# plots a heightmap using matplotlib
def plot_heightmap(heightmap,title):
    mapsize = len(heightmap)
    x = [i for i in range(mapsize)]
    y = [i for i in range(mapsize)]
    x, y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(10,4))
    ax = Axes3D(fig)
    #ax.plot_surface(x, y, heightmap, cmap=plt.cm.coolwarm, rcount=255, ccount=255)
    ax.plot_surface(x, y, heightmap, cmap=plt.cm.coolwarm, rcount=30, ccount=30, antialiased=False)
    plt.title(title)
    plt.axis([0,mapsize,0,mapsize])
    ax.set_zlim(0,256)
    ax.view_init(azim=0)
    ax.view_init(elev=50)


# saves the b&w image of the heightmap as a png
def save_image(heightmap,filename):
    Image.fromarray(heightmap).convert("L").save(filename,"PNG")


# Generates a 2d array heightmap using either Perlin noise or Worley noise
def generate_heightmap(mapsize, method):
    heightmap = np.zeros((mapsize, mapsize))
    if (method == "perlin"):
        offset = randint(0,(mapsize - 1))
        print("Generating Perlin Terrain...")
        for x in range(mapsize):
            for y in range(mapsize):
                heightmap[x][y] = noise.snoise2(x/(mapsize-1) + offset, y/(mapsize-1) + offset, octaves=4) * 128 + 128
    if (method == "worley"):
        print("Generating Worley Terrain...")
        NUM_PTS = 50
        N = 0
        wp_ys = np.zeros((NUM_PTS))
        wp_xs = np.zeros((NUM_PTS))
        for i in range(NUM_PTS):
            wp_ys[i] = randint(0, (mapsize-1))
            wp_xs[i] = randint(0, (mapsize-1))
        for x in range(mapsize):
            for y in range(mapsize):
                distances = [hypot(wp_xs[i] - x, wp_ys[i] - y) for i in range(NUM_PTS)]
                distances.sort()
                heightmap[x][y] = (mapsize-1) - (2 * (mapsize-1) * distances[N] / distances[-1] + (mapsize-1)/1.25 * distances[N+1]/distances[-1] + (mapsize-1)/1.5 * distances[N+2]/distances[-1]+ (mapsize-1)/1.75 * distances[N+3]/distances[-1])
    print("Complete!")
    return heightmap

### Functions for the hillshading method taken from https://github.com/dandrino/terrain-erosion-3-ways
def save_as_png(a, path):
  image = Image.fromarray(np.round(a * 255).astype('uint8'))
  image.save(path)

_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
    (0.00, (0.15, 0.3, 0.15)),
    (0.25, (0.3, 0.45, 0.3)),
    (0.50, (0.5, 0.5, 0.35)),
    (0.80, (0.4, 0.36, 0.33)),
    (1.00, (1.0, 1.0, 1.0)),
])

def hillshaded(a, land_mask=None, angle=270):
    if land_mask is None: land_mask = np.ones_like(a)
    ls = LightSource(azdeg=angle, altdeg=30)
    land = ls.shade(a, cmap=_TERRAIN_CMAP, vert_exag=10.0,
                  blend_mode='overlay')[:, :, :3]
    water = np.tile((0.25, 0.35, 0.55), a.shape + (1,))
    return lerp(water, land, land_mask[:, :, np.newaxis])

def load_from_file(path):
    result = np.load(path)
    if type(result) == np.lib.npyio.NpzFile:
        return (result['height'], result['land_mask'])
    else:
        return (result, None)

# Linear interpolation of `x` to `y` with respect to `a`
def lerp(x, y, a): return (1.0 - a) * x + a * y

def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)

def make_hillshaded_image(filename, output):
    height, land_mask = load_from_file(filename)
    save_as_png(hillshaded(height, land_mask=land_mask), output)



# Takes each value of `a` and offsets them by `delta`. Treats each grid point
# like a unit square.
def displace(a, delta):
    fns = {
        -1: lambda x: -x,
        0: lambda x: 1 - np.abs(x),
        1: lambda x: x,
    }
    result = np.zeros_like(a)
    for dx in range(-1, 2):
        wx = np.maximum(fns[dx](delta.real), 0.0)
    for dy in range(-1, 2):
        wy = np.maximum(fns[dy](delta.imag), 0.0)
        result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)
    return result

# Renormalizes the values of `x` to `bounds`
def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)
# Returns each value of `a` with coordinates offset by `offset` (via complex 
# values). The values at the new coordiantes are the linear interpolation of
# neighboring values in `a`.
def sample(a, offset):
    shape = np.array(a.shape)
    delta = np.array((offset.real, offset.imag))
    coords = np.array(np.meshgrid(*map(range, shape))) - delta

    lower_coords = np.floor(coords).astype(int)
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords 
    lower_coords %= shape[:, np.newaxis, np.newaxis]
    upper_coords %= shape[:, np.newaxis, np.newaxis]

    result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                     a[lower_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                lerp(a[upper_coords[1], lower_coords[0]],
                     a[upper_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                coord_offsets[1])
    return result

# Simple gradient by taking the diff of each cell's horizontal and vertical
# neighbors.
def simple_gradient(a):
  dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
  dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
  return 1j * dx + dy
##### Defines the actual hydraulic erosion algorithm
def erode(iterations):
    mapsize = 256
    full_width = 200
    shape = [mapsize] * 2
    cell_width = full_width / mapsize
    cell_area = cell_width ** 2

    rain_rate = .0008 * cell_width * cell_area
    evaporation_rate = .0005

    min_height_delta = .05
    gravity = 30
    gradient_sigma = .5

    sediment_capacity_constant = 50.0
    dissolving_rate = .25
    deposition_rate = .001

    terrain = generate_heightmap(mapsize, "perlin")
    #plot_heightmap(terrain, "before")
    # the amount of sediment that the water is carrying
    sediment = np.zeros_like(terrain)
    # the amount of water, carries the sediment
    water = np.zeros_like(terrain)
    # velocity of the water
    velocity = np.zeros_like(terrain)

    for i in range(iterations):
        print(i + 1, "/", iterations)
        # adds precipitation via a random distribution
        # *shape is an unpacking operator
        water += np.random.rand(*shape) * rain_rate

        # figure out where the water and sediment will be moving
        gradient = np.zeros_like(terrain, dtype="complex")
        gradient = simple_gradient(terrain)
        
        gradient = np.select([np.abs(gradient) < 1e-10],
                             [np.exp(2j * np.pi * np.random.rand(*shape))],
                             gradient)
        # renormalize the gradients
        gradient /= np.abs(gradient)

        # compute the difference between current height and the offset
        neighbor_height = sample(terrain, - gradient)
        height_delta = terrain - neighbor_height

        sediment_capacity = ((np.maximum(height_delta, min_height_delta) / cell_width) * velocity * water * sediment_capacity_constant)
        
        deposited_sediment = np.select(
                [
                  height_delta < 0, 
                  sediment > sediment_capacity,
                ], [
                  np.minimum(height_delta, sediment),
                  deposition_rate * (sediment - sediment_capacity),
                ],
                # If sediment <= sediment_capacity
                dissolving_rate * (sediment - sediment_capacity))

        # can't erode more than is currently there
        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        sediment -= deposited_sediment
        terrain += deposited_sediment
        sediment = displace(sediment, gradient)
        water = displace(water, gradient)

        velocity = gravity * height_delta / cell_width
        water *= 1- evaporation_rate

    save_image(terrain, "terrain.png")


erode(5)
