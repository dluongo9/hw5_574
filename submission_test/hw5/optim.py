from typing import Iterable

import numpy as np
from edugrad.optim import Optimizer, SGD
from edugrad.tensor import Tensor

# TODO: Replace with ref file

class Adagrad(Optimizer):

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2):
        super(Adagrad, self).__init__(params)
        self.lr = lr
        self._eps = 1e-7
        # initialize squared gradient history for each param
        for param in self.params:
            param._grad_hist = np.zeros(param.value.shape)

    def step(self):
        for param in self.params:
            # increment squared gradient history; param.grad contains the gradient
            gradient_hist = param._grad_hist
            gradient_hist += param.grad ** 2
            # get adjusted learning rate
            adjusted_learning_rate = self.lr / np.sqrt(gradient_hist + self._eps)
            # update parameters
            param.value -= adjusted_learning_rate * param.grad
        self._cur_step += 1
