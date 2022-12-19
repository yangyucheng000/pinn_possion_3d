import sys
sys.path.append("src/")

import core
import networks
from dataset import create_dataset
from argparse import ArgumentParser

import mindspore as ms
from mindspore import nn
from mindspore import numpy as ms_np
from mindelec.common import L2
import yaml


class WithEvalCell(nn.Cell):
    def __init__(self, model, solution):
        super().__init__(auto_prefix=False)
        self.model = model
        self.solution = solution

    def construct(self, x_domain, x_bc):
        y_pred_domain = self.model(x_domain)
        y_test_domain = self.solution(x_domain)

        y_pred_bc = self.model(x_bc)
        y_test_bc = self.solution(x_bc)
        return y_pred_domain, y_test_domain, y_pred_bc, y_test_bc


def test(dataset, model, target='possion_2d'):
    if target == 'possion_2d':
        solution = core.AnalyticSolution2D()
    elif target == 'possion_3d':
        solution = core.AnalyticSolution3D()
    else:
        raise ValueError

    eval_net = WithEvalCell(model, solution)
    eval_net.set_train(False)
    metric_domain = L2()
    metric_bc = L2()

    for x_domain, x_bc in dataset:
        y_pred_domain, y_test_domain, y_pred_bc, y_test_bc = eval_net(x_domain, x_bc)
        metric_domain.update(y_pred_domain.asnumpy(), y_test_domain.asnumpy())
        metric_bc.update(y_pred_bc.asnumpy(), y_test_bc.asnumpy())

    print("Relative L2 error (domain): {:.4f}".format(metric_domain.eval()))
    print("Relative L2 error (bc): {:.4f}".format(metric_bc.eval()))
    print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geometry', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--n_samps', default=5000, type=int)
    parser.add_argument('--batch_size', default=5000, type=int)
    args = parser.parse_args()
    
    # Create the dataset
    n_samps = args.n_samps
    batch_size = args.batch_size
    config_dataset = yaml.safe_load(open(args.geometry, "r"))
    config_dataset.update(n_samps_domain=n_samps, n_samps_bc=n_samps, batch_size=batch_size)
    target = config_dataset.pop('target')
    dataset = create_dataset(**config_dataset)
    print("Geometry: {}".format(config_dataset['shape_name']))

    # Create the model
    config_model = yaml.safe_load(open(args.model, "r"))
    model = networks.create_model(**config_model)
    checkpoint = ms.load_checkpoint(args.checkpoint)
    ms.load_param_into_net(model, checkpoint)
    
    # Test the model
    test(dataset, model, target)

