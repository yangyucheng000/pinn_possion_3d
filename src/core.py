import math

from mindspore import nn, ops
from mindspore import numpy as ms_np
from mindspore.ops import GradOperation


class GradWrtIdx(nn.Cell):
    def __init__(self, d_x, num_dims, i_dim):
        super().__init__()
        self.d_x = d_x
        self.num_dims = num_dims
        self.i_dim = i_dim
        
    def construct(self, x):
        return ms_np.hsplit(self.d_x(x), self.num_dims)[self.i_dim]


class Laplace(nn.Cell):
    def __init__(self, model, num_dims):
        super().__init__()
        d_x = GradOperation()(model)
        self.grads =[GradOperation()(GradWrtIdx(d_x, num_dims, i_dim)) for i_dim in range(num_dims)]
        self.num_dims = num_dims
            
    def construct(self, x):
        x_out = 0.
        for i_dim, d_xx in enumerate(self.grads):
            x_out += ms_np.hsplit(d_xx(x), self.num_dims)[i_dim]
        return x_out
    

class Possion(nn.Cell):
    def __init__(self, model, num_dims):
        super().__init__()
        self.laplace = Laplace(model, num_dims)
        self.prod = ops.ReduceProd(keep_dims=True)

    def construct(self, x):
        return self.laplace(x) + self.prod(ms_np.sin(4*math.pi*x), 1)


class Boundary(nn.Cell):
    def __init__(self, model, num_dims):
        super().__init__()
        self.model = model
        if num_dims == 2:
            self.solution = AnalyticSolution2D()
        elif num_dims == 3:
            self.solution = AnalyticSolution3D()
        else:
            raise ValueError

    def construct(self, x):
        return self.model(x) - self.solution(x)
        
    
class AnalyticSolution2D(nn.Cell):
    def construct(self, x):
        x_0, x_1 = ms_np.hsplit(ms_np.sin(4*math.pi*x), 2)
        return x_0*x_1/(32.*math.pi*math.pi)
    
    
class AnalyticSolution3D(nn.Cell):
    def construct(self, x):
        x_0, x_1, x_2 = ms_np.hsplit(ms_np.sin(4*math.pi*x), 3)
        return x_0*x_1*x_2/(48.*math.pi*math.pi)

