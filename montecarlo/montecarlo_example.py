#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:10:02 2023

@author: jozalen
"""

import numpy as np
import torch, time
from itertools import product
import torch.nn as nn
import functorch
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch-ready MCMC, variable number of walkers, fully parallelized

#torch.manual_seed(1)
d = 1
delta_MC = 1.
mean = torch.tensor(0.)
std = torch.tensor(1)
nwalkers = 10000
burn_in = 100

std_target = 3.
mean_target = 0.
rho = lambda x : ((1 / (std_target * torch.sqrt(2. * torch.tensor(torch.pi))))**d) * \
    torch.exp(-0.5*torch.sum(((x-mean_target) / std_target)**2, dim=1))

x0 = torch.randn(nwalkers, d)
t0 = time.time()
for _ in tqdm(range(burn_in)):
    ksi = torch.normal(mean=mean, std=std, size=(nwalkers, d))
    y = x0 + delta_MC * ksi
    r = rho(y) / rho(x0)
    p = torch.rand(nwalkers)
    comp = torch.gt(r, p).unsqueeze(1).expand(nwalkers, d)
    x0 = torch.where(comp, y, x0)
t1 = time.time()
print(f'Computation time: {round(t1-t0, 2)} s')

counts, bins = np.histogram(x0.numpy(), 50, density=True)
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.stairs(counts, bins)
plt.plot(np.linspace(-20, 20, nwalkers), rho(torch.linspace(-20, 20, nwalkers).unsqueeze(1)).numpy())








