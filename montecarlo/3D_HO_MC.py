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
import matplotlib.pyplot as plt

# My modules
import modules.neural_networks as neural_networks
from modules.plotters import minimisation_plots
from modules.dir_support import dir_support
from modules.aux_functions import train_loop, show_layers, split
from modules.loss_functions import HO_energy, overlap, HO_energy_trick

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
leap = 20
recompute = False
seed = 2
torch.manual_seed(seed)
pretrain = True
pre_epochs = 5000
path_pretrain = './pre.pt'

# Test mesh parameters
a = -8
b = 8

# MCMC parameters
d = 1
nwalkers = 1000
ntest = nwalkers
x0 = torch.randn(nwalkers, d, requires_grad=False)
burn_in = 1000
delta_MC = 0.5
mean = torch.tensor(0.)
std = torch.tensor(1)

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

optimizer = getattr(torch.optim, 'RMSprop')(params=psi_ann.parameters(),
                                lr=lr,
                                eps=epsilon,
                                alpha=alpha,
                                momentum=momentum)
    
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

h = (b - a) / (nwalkers - 1)
w_trpz = torch.ones(nwalkers**d) * (h**d)

################### PRETRAINING ###################
if pretrain:
    loss_fn = overlap
    N = nwalkers
    
    # Mesh
    x_pre = torch.linspace(a, b, N).requires_grad_()
    mesh_pre = torch.cartesian_prod(*(x_pre for _ in range(d))).reshape(N**d, d)
    
    # Target wave function
    target_1D = (1/torch.pi)**(1/4) * torch.exp(-(x_pre.detach()**2)/2)
    target_dD = torch.cartesian_prod(*(target_1D for _ in range(d))).reshape(N**d, 
                                                                             d)
    target_dD = torch.prod(target_dD, dim=1)
    
    ov_accum = []
    # Epoch loop
    for t in tqdm(range(pre_epochs)):
        ov, psi = loss_fn(model=psi_ann, # ov is really 1-<ANN|targ>^2
                        target=target_dD,
                        w_i=w_trpz,
                        train_data=mesh_pre)
        ov_accum.append(ov.item())
        optimizer.zero_grad()
        ov.backward()
        optimizer.step()
        
        if (t % 1000) == 0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            ax[0].plot([i for i in range(t + 1)], ov_accum)
            ax[1].plot(x_pre.detach().numpy(), psi.detach().numpy(), 
                       label='$\psi_{\mathrm{ANN}}$')
            ax[1].plot(x_pre.detach().numpy(), target_dD.numpy())
            ax[1].set_title('Pretraining', fontsize=17)
            plt.show()
    pretrain_state_dict = {'model':psi_ann.state_dict(),
                           'optimizer':optimizer.state_dict()}
    torch.save(pretrain_state_dict, path_pretrain)
        
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

if pretrain:
    optim_stdict = torch.load(path_pretrain)['optimizer']
    model_stdict = torch.load(path_pretrain)['model']
    psi_ann.load_state_dict(model_stdict)
    optimizer.load_state_dict(optim_stdict)
    psi_ann.train()

# Epoch loop
loss_fn = HO_energy
for t in tqdm(range(epochs)):
    # MCMC: we sample dD points distributed according to psi**2
    x0 = torch.randn(nwalkers, d, requires_grad=False)
    #print(x0)
    for _ in range(burn_in):
        ksi = torch.normal(mean=mean, std=std, size=(nwalkers, d))
        y = x0 + delta_MC * ksi
        r = (psi_ann(y) / psi_ann(x0)) ** 2
        p = torch.rand(nwalkers)
        comp = torch.gt(r, p).unsqueeze(1).expand(nwalkers, d)
        x0 = torch.where(comp, y, x0)
    
    mesh = x0.requires_grad_()
    x2_y2_z2 = torch.sum(mesh**2, dim=1).clone().detach()
    
    # Train loop: we compute psi on the MCMC mesh and we compute <Ĥ>
    E, psi = loss_fn(model=psi_ann,
                     train_data=mesh,
                     w_i=w_trpz,
                     x2_y2_z2=x2_y2_z2)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    E_accum.append(E.item())
    #K_accum.append(K.item())
    #U_accum.append(U.item())
    #overlap_accum.append(overlap.item())
    
    # Plotting
    if (((t+1) % leap) == 0 and periodic_plots) or t == epochs-1:
        if True:
            minimisation_plots(train_data=mesh,
                               x_axis=[i for i in range(t+1)], 
                               y_axis=E_accum,
                               K=K_accum,
                               U=U_accum,
                               y_exact=E_theoretical,
                               d=d,
                               x=x_test,
                               y=x_test,
                               z=psi, 
                               overlap=overlap_accum,
                               path_plot=path_plot,
                               show_last_plot=show_last_plot,
                               save=save_plot if t == epochs - 1 else False)
        
# Console feedback
print('\nModel trained!')
print('Total execution time:  {:6.2f} seconds ' \
      '(run on {})'.format(time.time() - start_time_all, device))


