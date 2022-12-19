from . import shapes
from . import rotating

import numpy as np
from mindelec.geometry import Geometry as BaseGeometry


class Geometry(BaseGeometry):
    def __init__(
        self, shape, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super().__init__(name, dim, coord_min, coord_max, dtype, sampling_config)
        self._shape = shape
        self.columns_dict = {}

    def _inside(self, points, strict=False):
        cond = self._shape.is_inside(points)
        if strict:
            cond |= self._on_boundary(points)
        return cond

    def _on_boundary(self, points):
        return self._shape.is_on_boundary(points)

    def _boundary_normal(self, points):
        return super()._boundary_normal(points)

    def sampling(self, geom_type="domain"):
        config = self.sampling_config
        if geom_type.lower() == "domain":
            self.columns_dict["domain"] = [self.name + "_domain_points"]
            if config.domain.random_sampling:
                data = self._shape.sample(config.domain.size).astype(self.dtype)
            else:
                raise NotImplementedError
        elif geom_type.lower() == "bc":
            self.columns_dict["BC"] = [self.name + "_BC_points"]
            if config.bc.random_sampling:
                data = self._shape.sample_boundary(config.bc.size).astype(self.dtype)
            else:
                raise NotImplementedError
        else:
            raise ValueError
        return data


class Simplex(Geometry):
    def __init__(self, name, vertices, boundary_type, dtype=np.float32, sampling_config=None):
        super().__init__(
            name=name,
            shape=shapes.Simplex(vertices, boundary_type),
            dim=2,
            coord_min=np.min(vertices, axis=0),
            coord_max=np.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config,
        )


class Pentagon(Geometry):
    def __init__(self, name, vertices, boundary_type, dtype=np.float32, sampling_config=None):
        super().__init__(
            name=name,
            shape=shapes.Pentagon(vertices, boundary_type),
            dim=2,
            coord_min=np.min(vertices, axis=0),
            coord_max=np.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config,
        )


class Tetrahedron(Geometry):
    def __init__(self, name, vertices, boundary_type, dtype=np.float32, sampling_config=None):
        super().__init__(
            name=name,
            shape=shapes.Simplex(vertices, boundary_type),
            dim=3,
            coord_min=np.min(vertices, axis=0),
            coord_max=np.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config
        )


class Cylinder(Geometry):
    def __init__(
        self, name, centre, radius, h_min, h_max, h_axis, boundary_type,
        dtype=np.float32, sampling_config=None):

        shape = rotating.Cylinder(centre, radius, h_min, h_max, h_axis, boundary_type)
        coord_min = np.append(np.asarray(centre) - np.asarray(radius), h_min)
        coord_max = np.append(np.asarray(centre) + np.asarray(radius), h_max)
        super().__init__(
            name=name,
            shape=shape,
            dim=3,
            coord_min=coord_min,
            coord_max=coord_max,
            dtype=dtype,
            sampling_config=sampling_config
        )


class Cone(Geometry):
    def __init__(
        self, name, centre, radius, h_min, h_max, h_axis, boundary_type,
        dtype=np.float32, sampling_config=None):

        shape = rotating.Cone(centre, radius, h_min, h_max, h_axis, boundary_type)
        coord_min = np.append(np.asarray(centre) - np.asarray(radius), h_min)
        coord_max = np.append(np.asarray(centre) + np.asarray(radius), h_max)
        super().__init__(
            name=name,
            shape=shape,
            dim=3,
            coord_min=coord_min,
            coord_max=coord_max,
            dtype=dtype,
            sampling_config=sampling_config
        )

