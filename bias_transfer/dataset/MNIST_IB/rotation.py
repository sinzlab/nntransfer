import numpy as np
from scipy import ndimage


def apply_rotation(source, target):
    angles = np.random.uniform(0, 360, source.shape[0])
    for i in range(source.shape[0]):
        source[i] = ndimage.rotate(source[i], angles[i], reshape=False, axes=(1, 2))
    return source, target