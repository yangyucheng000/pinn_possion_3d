import math

import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as ms_np
from mindspore.common import Parameter
from mindspore.common.initializer import initializer, HeUniform, Normal
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class OneCycleLR(LearningRateSchedule):
    def __init__(self, lr_max, total_steps, pct_start=0.3, init_factor=25, final_factor=10000.):
        super().__init__()
        self.lr_max = lr_max
        self.lr_init = lr_max/init_factor
        self.lr_final = lr_max/final_factor
        self.total_steps = total_steps
        self.step_mid = int(pct_start*total_steps)

    def construct(self, global_step):
        lr_max = self.lr_max
        lr_init = self.lr_init
        lr_final = self.lr_final
        total_steps = self.total_steps
        step_mid = self.step_mid
        
        global_step = global_step.astype(ms.float32)
        if global_step <= step_mid:
            cos_factor = 1 + ops.cos(math.pi*(global_step/step_mid - 1))
            return .5*(lr_max - lr_init)*cos_factor + lr_init
        else:
            cos_factor = 1 + ops.cos(math.pi*((global_step - step_mid)/(total_steps - step_mid)))
            return .5*(lr_max - lr_final)*cos_factor + lr_final


class ExpDeacy(nn.Cell):
    def __init__(self):
        super().__init__()
        self.factor = ms.Tensor(-1/(2*math.e))
        self.exp = ops.Exp()
        self.square = ops.Square()

    def construct(self, x):
        return x*self.exp(self.factor*self.square(x))


class LinearWN(nn.Cell):
    def __init__(
        self, in_features, out_features, weight_init='normal', g_init='normal', bias_init='zeros'):
        super().__init__()
        self.weight \
            = Parameter(initializer(weight_init, [in_features, out_features]), name='weight')
        self.sqrt_g = Parameter(initializer(g_init, out_features), name='sqrt_g')
        self.bias = Parameter(initializer(bias_init, out_features), name='bias')

    def construct(self, x):
        weight = self.weight
        weight = weight/ms_np.sqrt(ms_np.mean(weight*weight, axis=-1, keepdims=True))
        weight = self.sqrt_g*self.sqrt_g*weight
        return ms_np.matmul(x, weight) + self.bias


def create_model(input_size, base_neurons):
    return nn.SequentialCell(
        nn.Dense(input_size, 8*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(8*base_neurons, 4*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(4*base_neurons, 2*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(2*base_neurons, base_neurons, HeUniform(), activation=ExpDeacy()),
        LinearWN(base_neurons, 1, HeUniform(), Normal(sigma=.1))
    )

