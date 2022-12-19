import sys
sys.path.append("src/")

import os
import pickle
from argparse import ArgumentParser
import core
import networks
from dataset import create_dataset
from eval import WithEvalCell

import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as ms_np
from mindelec.data import Dataset
from mindelec.geometry import Rectangle, create_config_from_edict
from mindelec.architecture import MTLWeightedLossCell
from mindelec.common import L2
from easydict import EasyDict as edict
import yaml


class WithLossCell(nn.Cell):
    def __init__(self, eqn, bc, loss_fn):
        super().__init__(auto_prefix=False)
        self.eqn = eqn
        self.bc = bc
        self.loss_fn = loss_fn

        self.scale_domain = ms.Parameter(0., requires_grad=False)
        self.scale_bc = ms.Parameter(0., requires_grad=False)

    def construct(self, x_domain, x_bc):
        err_domain = self.eqn(x_domain)
        err_bc = self.bc(x_bc)
        return self.loss_fn((ms_np.mean(err_domain*err_domain), ms_np.mean(err_bc*err_bc)))


class TrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super().__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, x_domain, x_bc):
        loss = self.network(x_domain, x_bc)
        grads = self.grad(self.network, self.weights)(x_domain, x_bc)
        return loss, self.optimizer(grads)


def train(target, dataset, net_eqn, net_bc, model, learning_rate, n_epochs, save_name):
    loss_fn = MTLWeightedLossCell(num_losses=2)
    #loss_fn = BivariateNormalLoss()
    params = model.trainable_params() + loss_fn.trainable_params()
    opt = nn.Adam(params, learning_rate=learning_rate)
    
    with_loss_cell = WithLossCell(net_eqn, net_bc, loss_fn)
    train_net = TrainStep(with_loss_cell, opt)    

    if target == 'possion_2d':
        solution = core.AnalyticSolution2D()
    elif target == 'possion_3d':
        solution = core.AnalyticSolution3D()
    else:
        raise ValueError
    eval_net = WithEvalCell(model, solution)
    metric_domain = L2()
    metric_bc = L2()

    history_loss = []
    history_l2_domain = []
    history_l2_bc = []
    history = {
        'history_loss': history_loss,
        'history_l2_domain': history_l2_domain,
        'history_l2_bc': history_l2_bc,
    }
    for i_epoch in range(n_epochs):
        for x_domain, x_bc in dataset:
            loss, _ = train_net(x_domain, x_bc)
            loss = float(loss)
            history_loss.append(loss)

            y_pred_domain, y_test_domain, y_pred_bc, y_test_bc = eval_net(x_domain, x_bc)
            metric_domain.update(y_pred_domain.asnumpy(), y_test_domain.asnumpy())
            metric_bc.update(y_pred_bc.asnumpy(), y_test_bc.asnumpy())

        l2_domain = metric_domain.eval()
        l2_bc = metric_bc.eval()
        history_l2_domain.append(l2_domain)
        history_l2_bc.append(l2_bc)
        print("\ri_epoch={}, loss={:.2e}, l2(domain/BC)={:.4f}/{:.4f}".format(
            i_epoch, loss, l2_domain, l2_bc), end='')

        metric_domain.clear()
        metric_bc.clear()

    print("")

    ms.save_checkpoint(model, save_name)
    pickle.dump(history, open(save_name.replace(".ckpt", "_history.pickle"), "wb"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geometry', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--lr', default='src/config/learning_rate.yml', type=str)
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--save_dir', default='./')
    args = parser.parse_args()
    
    # Create the dataset
    config_dataset = yaml.safe_load(open(args.geometry, "r"))
    target = config_dataset.pop('target')
    dataset = create_dataset(**config_dataset)
    n_epochs = args.n_epochs
    steps_per_epoch = config_dataset['n_samps_domain']//config_dataset['batch_size']

    # Create the model
    config_model = yaml.safe_load(open(args.model, "r"))
    model = networks.create_model(**config_model)
    if target == 'possion_2d':
        num_dims = 2
    elif target == 'possion_3d':
        num_dims = 3
    else:
        raise ValueError
    net_eqn = core.Possion(model, num_dims)
    net_bc = core.Boundary(model, num_dims)

    # Create learning rate
    config_lr = yaml.safe_load(open(args.lr, "r"))
    learning_rate = networks.OneCycleLR(total_steps=steps_per_epoch*n_epochs, **config_lr)

    # Train the model
    save_name = "{}_{}.ckpt".format(target, config_dataset['shape_name'])
    save_name = os.path.join(args.save_dir, save_name)
    train(target, dataset, net_eqn, net_bc, model, learning_rate, n_epochs, save_name)

