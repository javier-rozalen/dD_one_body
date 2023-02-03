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
nwalkers = 10
d = 1
delta_MC = 1.
mean = torch.tensor(0.)
std = torch.tensor(1)
N = 10000
nsteps = 2.*nwalkers*N

rho = lambda x : ((1 / torch.sqrt(2. * torch.tensor(torch.pi)))**d) * torch.exp(-0.5*torch.sum(x**2, dim=1))

x0 = torch.randn(nwalkers, d)
mesh = []
n_accepted, c = 1, 0
t0 = time.time()
pbar = tqdm(total=N)
try:
    while n_accepted < N and c < nsteps - 1:
        n_accepted_prev = n_accepted
        ksi = torch.normal(mean=mean, std=std, size=(nwalkers, d))
        y = x0 + delta_MC * ksi
        r = rho(y) / rho(x0)
        p = torch.rand(nwalkers)
        comp = torch.gt(r, p).unsqueeze(1).expand(nwalkers, d)
        x0 = torch.where(comp, y, x0)
        mesh.append(x0)
        if c % (N / 10) == 0:
            n_accepted = int(torch.numel(torch.unique(torch.cat(mesh), dim=0)) / d) 
        c += 1
        pbar.update(n_accepted - n_accepted_prev)
    pbar.close()
except KeyboardInterrupt:
    pbar.close()

mesh_tensor = torch.cat(mesh)
mesh_unique = torch.unique(mesh_tensor, dim=0)
t1 = time.time()
print(f'Computation time: {round(t1-t0, 2)} s')

counts, bins = np.histogram(mesh_unique.numpy(), 50, density=True)
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.stairs(counts, bins)
plt.plot(mesh_unique.numpy(), rho(mesh_unique).numpy())








