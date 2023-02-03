#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#################### IMPORTS ####################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import torch, time, math
import numpy as np
from tqdm import tqdm
from itertools import product

# My modules
import modules.neural_networks as neural_networks
from modules.plotters import minimisation_plots
from modules.dir_support import dir_support
from modules.aux_functions import train_loop, show_layers, split
from modules.loss_functions import HO_energy

#################### ADJUSTABLE PARAMETERS ####################
# General parameters
device = 'cpu'
arch = '1sc'
nchunks_general = 1
which_chunk = int(sys.argv[1]) if nchunks_general != 1 else 0
save_model = False
save_plot = False
epochs = 250000
periodic_plots = True
show_last_plot = True
leap = 50
seed = 1
recompute = False
torch.manual_seed(seed)

# Test mesh parameters
a = -8
b = 8
ntest = 100

# MCMC parameters
d = 1
N = 5
nwalkers = 10
delta_MC = 1.
mean = torch.tensor(0.)
std = torch.tensor(1)
nsteps_MC = 2.*nwalkers*N
x0 = torch.randn(nwalkers, d)

loss_fn = HO_energy
E_theoretical = 0 + d/2

# Training hyperparameters
nhid = 100
actfun = 'Softplus'
optimizer = 'RMSprop'
lr = 0.1 # Use decimal notation 
epsilon = 1e-8
alpha = 0.9
momentum = 0.5

################### NEURAL NETWORK DEFINITION ###################   
# Neural Networks reference dictionary
net_arch_map = {'1sc': neural_networks.sc_1,
                '2sc': neural_networks.sc_2,
                '1sd': neural_networks.sd_1,
                '2sd': neural_networks.sd_2}
    
# ANN dimensions and initial weights
Nin = d
Nout = 1
W1 = torch.rand(nhid, Nin, requires_grad=True)*(-1.) 
B = torch.rand(nhid)*2. - torch.tensor(1.) 
W2 = torch.rand(Nout, nhid, requires_grad=True) 
Ws2 = torch.rand(1, nhid, requires_grad=True) 
Wd2 = torch.rand(1, nhid, requires_grad=True) 

# We load our NN and optimizer to the CPU (or GPU)
psi_ann = net_arch_map[arch](Nin, nhid, Nout, W1, 
                             Ws2, B, W2, Wd2, actfun).to(device)
try:
    optimizer = getattr(torch.optim, optimizer)(params=psi_ann.parameters(),
                                    lr=lr,
                                    eps=epsilon,
                                    alpha=alpha,
                                    momentum=momentum)
except TypeError:
    optimizer = getattr(torch.optim, optimizer)(params=psi_ann.parameters(),
                                    lr=lr)
    
#################### DIRECTORY SUPPORT ####################
path_steps = ['saved_models',
             'HO',
             'dD',
             'arch',
             'nhid',
             'optimizer',
             'actfun',
             'lr',
             'smoothing_constant',
             'momentum',
             'models/plots']
nsteps = range(len(path_steps))

################### MESH PREPARATION ###################
# Target wave function
x_test = torch.linspace(a, b, ntest)
test_data_dD = torch.cartesian_prod(*(x_test for _ in range(d))).reshape(ntest**d, d)
target_1D = (1/torch.pi)**(1/4) * torch.exp(-(x_test**2)/2)
target_dD = torch.cartesian_prod(*(target_1D for _ in range(d))).reshape(ntest**d, 
                                                                         d)
target_dD = torch.prod(target_dD, dim=1)


################### EPOCH LOOP ###################
start_time_all = time.time()
print(f'\nD = {d}, Arch = {arch}, Neurons = {nhid}, Actfun = {actfun}, ' \
            f'lr = {lr}, Alpha = {alpha}, Mu = {momentum}, ' \
              f'Seed = {seed}')
path_plot = f'saved_models/HO/{d}D/{arch}/nhid{nhid}/optim{optimizer}/' \
            f'{actfun}/lr{lr}/alpha{alpha}/mu{momentum}/plots/' \
            f'seed{seed}_epochs{epochs}'
# We store the energy data in lists for later plotting
E_accum = []
K_accum = []
U_accum = []
overlap_accum = []

for t in tqdm(range(epochs)):
    """
    # MCMC: we sample dD points distributed according to psi**2
    mesh = []
    n_accepted, c = 1, 0
    #pbar = tqdm(total=N)
    while n_accepted < N and c < nsteps_MC - 1:
        n_accepted_prev = n_accepted
        ksi = torch.normal(mean=mean, std=std, size=(nwalkers, d))
        y = x0 + delta_MC * ksi
        r = (psi_ann(y) / psi_ann(x0)) ** 2
        p = torch.rand(nwalkers)
        comp = torch.gt(r, p).unsqueeze(1).expand(nwalkers, d)
        x0 = torch.where(comp, y, x0)
        mesh.append(x0)
        if c % (N / 10) == 0:
            n_accepted = int(torch.numel(torch.unique(torch.cat(mesh), dim=0)) / d) 
        c += 1
        #pbar.update(n_accepted - n_accepted_prev)
    #pbar.close()
    mesh_tensor = torch.cat(mesh)
    mesh = torch.unique(mesh_tensor, dim=0).requires_grad_()
    """
    x = torch.linspace(a, b, N, requires_grad=True)
    mesh = torch.cartesian_prod(*(x for _ in range(d))).reshape(N**d, d)
    x2_y2_z2 = torch.sum(mesh**2, dim=1).clone().detach()
    
    # Train loop: we compute psi on the MCMC mesh and we compute <Ĥ>
    E, psi = train_loop(model=psi_ann, 
                              train_data=mesh,
                              x2_y2_z2=x2_y2_z2,
                              loss_fn=loss_fn, 
                              optimizer=optimizer)
    E_accum.append(E.item())
    #K_accum.append(K.item())
    #U_accum.append(U.item())
    #overlap_accum.append(overlap.item())
    
    # Plotting
    if (((t+1) % leap) == 0 and periodic_plots) or t == epochs-1:
        minimisation_plots(x_axis=[i for i in range(t+1)], 
                           y_axis=E_accum,
                           K=K_accum,
                           U=U_accum,
                           y_exact=E_theoretical,
                           d=d,
                           x=x_test,
                           y=x_test,
                           z=psi_ann(x_test.unsqueeze(1)), 
                           overlap=overlap_accum,
                           path_plot=path_plot,
                           show_last_plot=show_last_plot,
                           save=save_plot if t == epochs - 1 else False)
        
# Console feedback
print('\nModel trained!')
print('Total execution time:  {:6.2f} seconds ' \
      '(run on {})'.format(time.time() - start_time_all, device))


