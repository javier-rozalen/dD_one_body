"""
Guide to compute first and second order derivatives in pytorch.
"""
import numpy as np
import torch, time
from itertools import product
import functorch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchviz import make_dot

from modules import neural_networks

x = torch.randn(2, 2, requires_grad=True)

# Scalar outputs
out = x.sum()  # Size([])
batched_grad = torch.arange(3)  # Size([3])
grad, = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)

# loop approach
grads = torch.stack(([torch.autograd.grad(out, x, torch.tensor(a))[0] for a in range(3)]))











