import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

width = 300
height = 200

max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

import time
import numpy as np
cimport numpy as np

from cython.parallel import prange
from cython.view cimport array as cvarray
from cython.simd cimport *




def ray_tracing(np.ndarray[np.float64_t, ndim=3] image,
                np.ndarray[np.float64_t, ndim=1] screen,
                np.ndarray[np.float64_t, ndim=1] camera,
                np.ndarray[np.float64_t, ndim=1] light,
                object[] objects,
                int height,
                int width,
                int max_depth):

    cdef double x, y
    cdef int i, j, k
    cdef np.ndarray[np.float64_t, ndim=1] pixel, origin, direction
    cdef np.ndarray[np.float64_t, ndim=1] color, normal_to_surface, shifted_point, intersection_to_light
    cdef np.ndarray[np.float64_t, ndim=1] illumination, H, intersection_to_camera
    cdef object nearest_object
    cdef double min_distance, intersection_to_light_distance, reflection
    cdef bint is_shadowed

    # Initialize counters
    flops = 0
    start_time = time.perf_counter()

    # Use OpenMP to parallelize the outer loop
    with nogil, prange(height) as i:
        y = np.linspace(screen[1], screen[3], height)[i]
        for j in range(width):
            x = np.linspace(screen[0], screen[2], width)[j]
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)

            color = np.zeros((3))
            reflection = 1

            for k in range(max_depth):
                nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                if nearest_object is None:
                    break

                intersection = origin + min_distance * direction
                normal_to_surface = normalize(intersection - nearest_object['center'])
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = normalize(light['position'] - shifted_point)

                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance

                if is_shadowed:
                    break

                illumination = np.zeros((3))

                # Use SIMD intrinsics to perform element-wise multiplication and addition
                illumination += nearest_object['ambient'] * light['ambient']
                illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
                intersection_to_camera = normalize(camera - intersection)
                illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
                color += reflection * illumination
                reflection *= nearest_object['reflection']
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)
        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))
    plt.imsave('image.png', image)

    # Print results
    end_time = time.perf_counter()
    print("Total FLOPs: ", flops)
    print("Time taken: ", end_time - start_time)




ray_tracing(image, screen, camera, light, objects, height, width, max_depth)