import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2


class GELU(nn.Module):
    """
    Gaussian Error Linear Units (GELU) - https://arxiv.org/abs/1606.08415
    GELU(x) = 0.5*x*(1 + tanh[sqrt(2/π)(x + 0.044715*x^3)]) = x*σ(1.702*x)
    """
    def forward(self, x):
        return (0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0) / torch.pi) * (x + 0.044715 * torch.pow(x, 3.0)))))


class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) - https://arxiv.org/abs/1803.08375
    ReLU(x) = max(0, x)
    """
    def forward(self, x):
        return torch.max(torch.tensor(0.0), x)


class LayerNorm(nn.Module):
    """
    Layer Normalization - https://arxiv.org/abs/1607.06450
    h(x) = γ ⊙ (x - μ(x)) / (σ(x) + ε) + β
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.γ = nn.Parameter(torch.ones(features))
        self.β = nn.Parameter(torch.zeros(features))
        self.ε = eps

    def forward(self, x: torch.FloatTensor):
        μ = x.mean(dim=-1, keepdim=True)
        σ = x.std(dim=-1, keepdim=True)
        return self.γ * (x - μ) / (σ + self.ε) + self.β
    

class BatchNorm(nn.Module):
    """
    Batch Normalization - https://arxiv.org/abs/1502.03167
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.ε = eps
        self.γ = nn.Parameter(torch.ones(num_features))
        self.β = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('μ_running', torch.zeros(num_features))
        self.register_buffer('σ_running', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Calculate batch statistics during training
            μ = x.mean(dim=(0, 2, 3), keepdim=True)
            σ = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Update running statistics using momentum
            self.μ_running = (1 - self.momentum) * self.μ_running + self.momentum * μ.squeeze()
            self.σ_running = (1 - self.momentum) * self.σ_running + self.momentum * σ.squeeze()
        else:
            # Use running statistics during inference
            μ = self.μ_running.view(1, -1, 1, 1)
            σ = self.σ_running.view(1, -1, 1, 1)

        # Normalize input
        x_norm = (x - μ) / torch.sqrt(σ + self.ε)
        # Scale and shift
        x_scaled = self.γ.view(1, -1, 1, 1) * x_norm + self.β.view(1, -1, 1, 1)
        return x_scaled


class Dropout(nn.Module):
    """
    Dropout - https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    Mask layer outputs with probability p (default p=0.5)
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask = torch.empty_like(x).bernoulli_(1 - self.p)

        if self.inplace:
            x *= mask
            return x
        return x * mask


class Linear(nn.Module):
    """
    Linear transfomration. Values initalized from U(-√k, √k) where k=1/in_features
    y = Wx + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=torch.sqrt(5)) # Kaiming He initalization - https://arxiv.org/abs/1502.01852
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / torch.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, input):
        output = input.matmul(self.W.t())
        if self.b is not None:
            output += self.b
        return output

