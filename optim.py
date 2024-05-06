import torch

class AdamW:
    """
    Adam Optimizer - https://arxiv.org/abs/1412.6980
    PyTorch implements an optimized version, see Algorithm 2 in https://arxiv.org/abs/1711.05101 for discussion
    """
    def __init__(self, parameters, lr=0.001, β1=0.9, β2=0.999, ε=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.β1 = β1
        self.β2 = β2
        self.ε = ε
        self.t = 0
        self.m = [torch.zeros_like(param) for param in self.parameters]
        self.v = [torch.zeros_like(param) for param in self.parameters]

    def step(self, gradients):
        self.t += 1
        for i, param in enumerate(self.parameters):
            self.m[i] = self.β1 * self.m[i] + (1 - self.β1) * gradients[i]
            self.v[i] = self.β2 * self.v[i] + (1 - self.β2) * (gradients[i] ** 2)
            m_hat = self.m[i] / (1 - self.β1 ** self.t)
            v_hat = self.v[i] / (1 - self.β2 ** self.t)
            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.ε)
            param.data -= update


class SGD:
    """
    TODO
    """