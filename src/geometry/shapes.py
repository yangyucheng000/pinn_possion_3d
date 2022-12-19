import math
from itertools import combinations
from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    def __init__(self, volume, boundary=None):
        self._volume = volume
        self._boundary = boundary

    @property
    def volume(self):
        """Volume of the shape (length for 1D shape and area for 2D shape)."""
        return self._volume

    @abstractmethod
    def is_inside(self, pts):
        """Check if the points is inside the shape.

        Should return a flattened array.
        """
        pass

    def is_on_boundary(self, pts):
        if self._boundary is None:
            raise ValueError("The shape has no boundary.")
        else:
            return self._boundary.is_inside(pts)

    @abstractmethod
    def sample(self, num_samps):
        """Uniformly sample points in the shape."""
        pass

    def sample_boundary(self, num_samps):
        if self._boundary is None:
            raise ValueError("The shape has no boundary.")
        else:
            return self._boundary.sample(num_samps)


class Union(Shape):
    def __init__(self, shapes, boundary=None, has_overlap=False, use_probs=True):
        probs = np.array([sh.volume for sh in shapes])
        volume = sum(probs)
        if use_probs:
            probs /= volume
        else:
            probs = [1./len(shapes)]*len(shapes)

        super().__init__(volume, boundary)
        self._probs = probs
        self._shapes = shapes
        self._has_overlap = has_overlap

    def is_inside(self, pts):
        return np.any([sh.is_inside(pts) for sh in self._shapes], axis=0)

    def sample(self, num_samps):
        num_each = np.random.multinomial(num_samps, self._probs)
        samps = np.vstack([sh.sample(num) for num, sh in zip(num_each, self._shapes)])
        np.random.shuffle(samps)

        if self._has_overlap:
            raise NotImplementedError

        return samps


class Segment(Shape):
    def __init__(self, vertices):
        x0, y0, x1, y1 = np.ravel(vertices)
        self._slope = (y1 - y0)/(x1 - x0)
        self._intercpet = (y0*x1 - y1*x0)/(x1 - x0)
        self._x_min = min(x0, x1)
        self._x_max = max(x0, x1)
        super().__init__(volume=math.sqrt((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1)))

    def is_inside(self, pts):
        x = pts[:, 0]
        return (x >= self._x_min) & (x <= self._x_max) & self.is_on(pts)

    def is_on(self, pts):
        """Check if ``x`` is on the line defined by the segment."""
        x, y = pts.T
        return np.isclose(y, self.line_equation(x))

    def sample(self, num_samps):
        x = self._x_min + (self._x_max - self._x_min)*np.random.rand(num_samps, 1)
        return np.hstack([x, self.line_equation(x)])

    def line_equation(self, x):
        return self._intercpet + self._slope*x


class Simplex(Shape):
    """Simplex (triangle in 2D and tetrahedron in 3D)."""
    def __init__(self, vertices, boundary_type='none'):
        vertices = self._validate(vertices)
        # Compute volume
        volume = .5*abs(np.linalg.det(np.hstack([np.ones([len(vertices), 1]), vertices])))
        # Prepare boundary
        num_vertices = len(vertices)
        if num_vertices == 3:
            cls = Segment
        elif num_vertices == 4:
            cls = Triangle3D
        else:
            raise ValueError
        shapes = [cls(vertices[list(indices)])
            for indices in combinations(range(num_vertices), num_vertices - 1)]
        boundary = derive_union_boundary(shapes, boundary_type)
        super().__init__(volume, boundary)
        # Prepare transform matrix
        self._x0 = vertices[0]
        matrix = vertices[1:] - vertices[0]
        if np.linalg.det(matrix) < 0:
            tmp = matrix[0].copy()
            matrix[0] = matrix[1]
            matrix[1] = tmp
        self._matrix = matrix
        self._matrix_inv = np.linalg.inv(matrix)

    def is_inside(self, pts):
        pts = self.inverse_transform(pts)
        return np.all(pts > 0, axis=1) & (pts.sum(axis=1) < 1.)

    def sample(self, num_samps):
        return self.transform(self.sample_unit_triangle(num_samps))

    def sample_unit_triangle(self, num_samps):
        samps = np.random.rand(num_samps, 2)
        cond = np.sum(samps, axis=1) > 1
        samps[cond] = 1 - samps[cond][:, [1, 0]]

        if len(self._x0) == 3:
            # Notice the area of the triangle at z is .5*(1 - z)^2
            z = 1 - np.cbrt(np.random.rand(num_samps, 1))
            samps = np.hstack([(1 - z)*samps, z])

        return samps

    def transform(self, pts):
        """Transfrom the unit triangle into the given triangle."""
        return np.matmul(pts, self._matrix) + self._x0

    def inverse_transform(self, pts):
        """Transfrom the given triangle into the unit triangle."""
        return np.matmul(pts - self._x0, self._matrix_inv)

    def _validate(self, vertices):
        vertices = np.asarray(vertices).copy()
        if vertices.shape == (3, 2):
            if Segment(vertices[:2]).is_on(vertices[2]):
                raise ValueError("Three points lie on a line.")
        elif vertices.shape == (4, 3):
            pass
        else:
            raise ValueError
        return vertices


class Triangle3D(Shape):
    def __init__(self, vertices):
        vertices = self._validate(vertices)
        # Check if the given triangle can be reduced into 2D.
        dim_reduced = None
        for i_col in range(vertices.shape[1]):
            if np.allclose(vertices[:, i_col], vertices[0, i_col]):
                dim_reduced = i_col
                val_reduced = vertices[0, i_col]
                break

        if dim_reduced is None:
            # Compute area
            cross_product = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            volume = .5*abs(np.linalg.norm(cross_product, ord=2))
            super().__init__(volume)
            # Prepare transform matrix
            matrix = vertices
            if np.linalg.det(matrix) < 0:
                matrix = np.asarray([matrix[1], matrix[0], matrix[2]])
            self._matrix = matrix
            self._matrix_inv = np.linalg.inv(matrix)
            #
            self._tri_reduced = None
        else:
            self._tri_reduced = Simplex(
                np.delete(vertices, dim_reduced, axis=1), boundary_type='none')
            self._dim_reduced = dim_reduced
            self._val_reduced = val_reduced
            super().__init__(self._tri_reduced.volume)

    def is_inside(self, pts):
        if self._tri_reduced is None:
            pts = self.inverse_transform(pts)
            return np.all((pts > 0.) & (pts < 1.), axis=1) & np.isclose(pts.sum(axis=1), 1.)
        else:
            pts_reduced = np.delete(pts, self._dim_reduced, axis=1)
            return self._tri_reduced.is_inside(pts_reduced) \
                & np.isclose(pts[:, self._dim_reduced], self._val_reduced)

    def sample(self, num_samps):
        if self._tri_reduced is None:
            return self.transform(self.sample_unit_triangle(num_samps))
        else:
            samps_reduced = self._tri_reduced.sample(num_samps)
            samps = np.insert(samps_reduced, self._dim_reduced, self._val_reduced, axis=1)
            return samps

    def sample_unit_triangle(self, num_samps):
        """Uniformly sample points in the plane defined by (0, 0, 1), (1, 0, 0)
        and (0, 1, 0).

        This is equivalent to sample from a flat dirichlet distribution.
        """
        samps = np.log(np.random.rand(num_samps, 3))
        samps /= samps.sum(axis=1, keepdims=True)
        return samps

    def transform(self, pts):
        """Transfrom the unit triangle into the given triangle."""
        return np.matmul(pts, self._matrix)

    def inverse_transform(self, pts):
        """Transfrom the given triangle into the unit triangle."""
        return np.matmul(pts, self._matrix_inv)

    def _validate(self, vertices):
        vertices = np.asarray(vertices).copy()
        assert vertices.shape == (3, 3)
        # TODO Check the cases that three points lie on a line
        return vertices


class Pentagon(Union):
    """Pentagon.

    Aussme that the adjacent input vertices form en edge of the Pentagon. The
    code does not check the case that three points lie on a line.
    """
    def __init__(self, vertices, boundary_type='none'):
        vertices = self._validiate(vertices)
        shapes = self._create_triangles(vertices)
        segments = [Segment([vertices[idx], vertices[self._c_idx(idx + 1)]])
            for idx in range(len(vertices))]
        boundary = derive_union_boundary(segments, boundary_type)
        super().__init__(shapes, boundary)

    def _create_triangles(self, vertices):
        """Divide the given pentagon into three triangles."""
        def compute_angle(vec0, vec1):
            """Compute the angle of two vectors."""
            cos_t = np.inner(vec0, vec1)/(np.linalg.norm(vec0, 2)*np.linalg.norm(vec1, 2))
            return math.acos(cos_t)

        def area(vertices_tri):
            """Compute the area of a triangle."""
            x0, y0, x1, y1, x2, y2 = np.ravel(vertices_tri)
            return .5*abs(x2*y0 + x1*y2 + x0*y1 - x0*y2 - x2*y1 - x1*y0)

        # Check if the given pentagon is convex
        num_angles = 5
        angles = [0.]*num_angles
        for i_v in range(num_angles):
            vec0 = vertices[self._c_idx(i_v - 1)] - vertices[i_v]
            vec1 = vertices[self._c_idx(i_v + 1)] - vertices[i_v]
            angles[i_v] = compute_angle(vec0, vec1)

        theta_tot = sum(angles)
        if math.isclose(theta_tot, 3*math.pi):
            idx_start = 0
        else:
            for i_theta, theta in enumerate(angles):
                if math.isclose(theta_tot - 2*theta + 2*math.pi, 3*math.pi):
                    idx_start = i_theta
                    break

        vertices_tri = [
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start + 1)],
                vertices[self._c_idx(idx_start + 2)]]),
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start - 1)],
                vertices[self._c_idx(idx_start - 2)]]),
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start + 2)],
                vertices[self._c_idx(idx_start - 2)]]),
        ]
        return [Simplex(ver, boundary_type='none') for ver in vertices_tri]

    def _c_idx(self, idx):
        """Get cyclic indices."""
        idx_max = 5
        if idx >= idx_max:
            return idx - idx_max
        elif idx < 0:
            return idx + idx_max
        else:
            return idx

    def _validiate(self, vertices):
        vertices = np.asarray(vertices).copy()
        assert len(vertices) == 5
        return vertices


def derive_union_boundary(shapes, boundary_type):
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
    return boundary

