from .shapes import Shape, Union, derive_union_boundary

import math
from abc import abstractmethod

import numpy as np


class RotatingShape(Shape):
    """Base class for rotating geometry. The geometry should be defined by the
    following parameters:

    Args:
        centre: Origin.
        radius: Radius.
        h_min: Minimum height.
        h_max: Maximum height.
        h_axis: Height axis.
    """
    def __init__(self, centre, radius, h_min, h_max, h_axis, is_surface, volume, boundary=None):
        super().__init__(volume, boundary)
        self._centre = centre
        self._radius = radius
        self._h_min = h_min
        self._h_max = h_max
        self._height = h_max - h_min
        self._h_axis = h_axis
        self._is_surface = is_surface

    def is_inside(self, pts):
        pts_disk, pts_height = self.split_height(pts)
        r_to_centre = self.radius_to_centre(pts_disk)
        r_at_height = self._radius*self.radius_at_height(pts_height)

        if pts_height is None:
            # Assume that the shape is a 2D disk
            is_within_height = True
        elif math.isclose(self._h_min, self._h_max):
            # Assuem that the shape is a 3D disk
            is_within_height = np.isclose(pts_height, self._h_min)
        else:
            is_within_height = (pts_height > self._h_min) & (pts_height < self._h_max)

        if self._is_surface:
            return np.isclose(r_to_centre, r_at_height) & is_within_height
        else:
            return (r_to_centre < r_at_height) & is_within_height

    def sample(self, num_samps):
        samps_height = self._h_min + self._height*self.sample_height(num_samps)
        samps_radius = self._radius*np.atleast_1d(self.radius_at_height(samps_height))[:, None]
        if not self._is_surface:
            samps_radius = samps_radius*np.sqrt(np.random.rand(num_samps, 1))
        samps_disk = self._centre + samps_radius*sample_unit_sphere(num_samps, 2)

        # Check the special case where the shape is a disk
        if self._h_axis is None:
            samps = samps_disk
        else:
            samps = self.insert_height(samps_disk, samps_height)

        return samps

    @abstractmethod
    def sample_height(self, num_samps):
        """Sample normalised height.

        - Should return a flattened array.
        """
        pass

    @abstractmethod
    def radius_at_height(self, height):
        """Sample normalised radius at the given height.

        - Assume the input is a flattened array, float or None.
        - Should return a flattened array or a scalar.
        """
        pass

    def radius_to_centre(self, pts, keepdims=False):
        return np.linalg.norm(pts - self._centre, ord=2, axis=1, keepdims=keepdims)

    def split_height(self, pts):
        if self._h_axis is None:
            return pts, None
        else:
            pts_disk = np.delete(pts, self._h_axis, axis=1)
            pts_height = pts[:, self._h_axis]
            return pts_disk, pts_height

    def insert_height(self, pts, height):
        return np.insert(pts, self._h_axis, height, axis=1)


class Disk(RotatingShape):
    def __init__(self, centre, radius, height=0., h_axis=None, has_boundary=False):
        area = math.pi*radius*radius
        if has_boundary:
            boundary = Ring(centre, radius, height, h_axis)
        else:
            boundary = None
        super().__init__(
            centre, radius, height, height, h_axis,
            is_surface=False, volume=area, boundary=boundary)

    def sample_height(self, num_samps):
        return self._h_min

    def radius_at_height(self, height):
        return 1.


class Ring(RotatingShape):
    def __init__(self, centre, radius, height=0., h_axis=None):
        length = 2*math.pi*radius
        super().__init__(
            centre, radius, height, height, h_axis,
            is_surface=True, volume=length, boundary=None)

    def sample_height(self, num_samps):
        return self._h_min

    def radius_at_height(self, height):
        return 1.


class Cylinder(RotatingShape):
    def __init__(self, centre, radius, h_min, h_max, h_axis, boundary_type='none'):
        volume = math.pi*radius*radius*(h_max - h_min)
        shapes = [
            Disk(centre, radius, h_min, h_axis),
            CylinderSurface(centre, radius, h_min, h_max, h_axis),
            Disk(centre, radius, h_max, h_axis),
        ]
        boundary = derive_union_boundary(shapes, boundary_type)
        if boundary_type == 'none':
            boundary = None
        else:
            if boundary_type == 'uniform':
                use_probs = True
            elif boundary_type == 'unweighted':
                use_probs = False
            else:
                raise ValueError
            boundary = Union(shapes, use_probs=use_probs)
        super().__init__(
            centre, radius, h_min, h_max, h_axis,
            is_surface=False, volume=volume, boundary=boundary)

    def sample_height(self, num_samps):
        return np.random.rand(num_samps)

    def radius_at_height(self, height):
        return 1.


class CylinderSurface(RotatingShape):
    def __init__(self, centre, radius, h_min, h_max, h_axis):
        area = 2*math.pi*radius*(h_max - h_min)
        super().__init__(
            centre, radius, h_min, h_max, h_axis,
            is_surface=True, volume=area, boundary=None)

    def sample_height(self, num_samps):
        return np.random.rand(num_samps)

    def radius_at_height(self, height):
        return 1.


class Cone(RotatingShape):
    def __init__(self, centre, radius, h_min, h_max, h_axis, boundary_type='none'):
        volume = math.pi*radius*radius*(h_max - h_min)/6.
        shapes = [
            ConeSurface(centre, radius, h_min, h_max, h_axis),
            Disk(centre, radius, h_min, h_axis),
        ]
        boundary = derive_union_boundary(shapes, boundary_type)
        super().__init__(
            centre, radius, h_min, h_max, h_axis,
            is_surface=False, volume=volume, boundary=boundary)

    def sample_height(self, num_samps):
        return 1 - np.cbrt(np.random.rand(num_samps))

    def radius_at_height(self, height):
        return 1 - (height - self._h_min)/self._height


class ConeSurface(RotatingShape):
    def __init__(self, centre, radius, h_min, h_max, h_axis):
        area = math.pi*radius*(h_max - h_min)
        super().__init__(
            centre, radius, h_min, h_max, h_axis,
            is_surface=True, volume=area, boundary=None)

    def sample_height(self, num_samps):
        return 1 - np.sqrt(np.random.rand(num_samps))

    def radius_at_height(self, height):
        return 1 - (height - self._h_min)/self._height


def sample_unit_sphere(num_samps, num_dims):
    """Sample points on a unit sphere (ring for 2D)"""
    assert num_dims > 1
    samps = np.random.randn(num_samps, num_dims)
    samps /= np.linalg.norm(samps, ord=2, axis=1, keepdims=True)
    return samps

