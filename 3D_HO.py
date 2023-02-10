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
from torchviz import make_dot
from modules.ada_hessian import AdaHessian

# My modules
import modules.neural_networks as neural_networks
from modules.plotters import minimisation_plots, pretraining_plots
from modules.aux_functions import train_loop, show_layers, split, dir_support
from modules.loss_functions import HO_energy, HO_energy_2der, overlap

#################### ADJUSTABLE PARAMETERS ####################
# General parameters
device = 'cpu' 
net_archs = ['1sc']
nchunks_general = 1
which_chunk = int(sys.argv[1]) if nchunks_general != 1 else 0
seed = 1
torch.manual_seed(seed)
recompute = True
nders = 1

# Traning params
save_model = False
save_plot = False
epochs = 500
periodic_plots = True
show_last_plot = True
leap = 200

# Pretraining params
pretrain = True
pre_epochs = 2000
pre_leap = 500
path_pretrain = './pre.pt'

# Computation graph prameters
computation_graph = True
show_attrs = False
show_saved = False
retain_graph = True
create_graph = True
cg_name = f'cg_retain{retain_graph}_create{create_graph}'

# Mesh parameters
N = 50 # points per dimension
a = -5
b = 5
h = (b-a) / (N)

loss_fn = HO_energy

# Training hyperparameters
dims = [3]
hidden_neurons = [300]
actfuns = ['Sigmoid']
optimizers = ['RMSprop']
learning_rates = [0.05] # Use decimal notation 
epsilon = 1e-8
smoothing_constants = [0.9]
momenta = [0.0]

################### MESH PREPARATION ###################
"""
#mesh = torch.reshape(mesh, (N**d, d)) # (N**d, d)
print(f'{d}D cartesian product mesh:\n{mesh}, {mesh.shape}\n')
y = torch.meshgrid([x for _ in range(d)])
print(f'{d}D meshgrid mesh:\n{y}, {y[0].shape}\n')
"""

################### INTEGRATION ###################
"""
w_trpz = torch.ones(*(N for _ in range(d)))*torch.tensor(h)
print(f'Final weight matrix:\n {w_trpz}, {w_trpz.shape}\n')
print(f'Initial weight matrix:\n {w_trpz}, {w_trpz.shape}\n')
src = h/2
j2 = [N for _ in range(d - 1)]
j2.append(2)
index = torch.tensor([0, N - 1]).expand(j2)
w_trpz = w_trpz.scatter(d-1, index, src) # ()
"""

################### NEURAL NETWORK DEFINITION ###################   
# Neural Networks reference dictionary
net_arch_map = {'1sc': neural_networks.sc_1,
                '2sc': neural_networks.sc_2,
                '1sd': neural_networks.sd_1,
                '2sd': neural_networks.sd_2}
    
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

################### EPOCH LOOP ###################
start_time_all = time.time()
if nchunks_general > len(dims):
    nchunks = len(dims)
else:
    nchunks = nchunks_general
if which_chunk >= nchunks:
    sys.exit()
for d in split(dims, nchunks)[which_chunk]:
    for arch, nhid, optim, actfun, lr, alpha, mom in product(net_archs,
                                                             hidden_neurons,
                                                             optimizers, actfuns,
                                                             learning_rates,
                                                             smoothing_constants,
                                                             momenta):            
        print(f'\nD = {d}, Arch = {arch}, Neurons = {nhid}, Actfun = {actfun}, ' \
                    f'lr = {lr}, Alpha = {alpha}, Mu = {mom}, ' \
                      f'Seed = {seed}')
        # Directory support
        for _ in nsteps:
            path_steps_models = ['saved_models',
                                 'HO',
                                 f'{d}D',
                                 f'{arch}',
                                 f'nhid{nhid}',
                                 f'optim{optim}',
                                 f'{actfun}',
                                 f'lr{lr}', 
                                 f'alpha{alpha}', 
                                 f'mu{mom}',
                                 'models']
            path_steps_plots = ['saved_models',
                                'HO',
                                f'{d}D',
                                f'{arch}',
                                f'nhid{nhid}',
                                f'optim{optim}',
                                f'{actfun}',
                                f'lr{lr}', 
                                f'alpha{alpha}', 
                                f'mu{mom}',
                                'plots']
            if save_model:
                dir_support(path_steps_models)
            if save_plot:
                dir_support(path_steps_plots) 
                
        # If the parameter 'recompute' is set to 'False' and the model
        # analyzed in the current iteration had been already trained, we
        # skip this model.
        saved_models_dir = '/'.join(path_steps_models) + '/'
        name_without_dirs = f'seed{seed}_epochs{epochs}.pt'
        try:
            if (recompute == False and 
                name_without_dirs in os.listdir(saved_models_dir)): 
                print(f'Skipping already trained model {name_without_dirs}...')
                continue
        except FileNotFoundError:
            pass
        
        ################### MESH PREPARATION ###################
        # We construct the mesh. It has dimension (d*N^d), because there are N^d 
        # points and each one is d-dimensional.
        x = torch.linspace(a, b, N, requires_grad=False)
        mesh = torch.cartesian_prod(*(x for _ in range(d))).reshape(N**d, d)
        mesh.requires_grad_() # adding the 'requires_grad' property here makes the 
                              # computation graph simpler
        x2_y2_z2 = torch.sum(mesh**2, dim=1).clone().detach()
        
        ################### INTEGRATION ###################
        # We construct the weight matrix. It has dimension (N^d)
        #w_trpz = torch.ones(N**d) * (h**d)
        w_trpz = h**d
        E_theoretical = 0 + d/2
        
        # Target wave function
        x_ov = x.detach().clone()
        target_1D = (1/torch.pi)**(1/4) * torch.exp(-(x_ov**2)/2)
        NN = len(x_ov)
        target_dD = torch.cartesian_prod(*(target_1D for _ in range(d))).reshape(NN**d, d)
        target_dD = torch.prod(target_dD, dim=1)
                
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
            if optim == 'AdaHessian':
                optimizer = AdaHessian(params=psi_ann.parameters())
            else:
                optimizer = getattr(torch.optim, optim)(params=psi_ann.parameters(),
                                                lr=lr,
                                                eps=epsilon,
                                                alpha=alpha,
                                                momentum=mom)
        except TypeError:
            optimizer = getattr(torch.optim, optim)(params=psi_ann.parameters(),
                                            lr=lr)
            
        # Pretraining 
        if pretrain:
            ov_accum = []
            # Epoch loop
            for t in tqdm(range(pre_epochs)):
                ov, loss, psi = overlap(model=psi_ann, # ov is really 1-<ANN|targ>^2
                                target_dD=target_dD,
                                w_i=w_trpz,
                                train_data=mesh)
                ov_accum.append(ov.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (t % pre_leap) == 0:
                    pretraining_plots(d=d,
                                      mesh_1d=x_ov,
                                      mesh=mesh,
                                      psi_normalized=psi, 
                                      target_dD=target_dD,
                                      nepochs=t,
                                      overlap_accum=ov_accum)
                        
            pretrain_state_dict = {'model':psi_ann.state_dict(),
                                   'optimizer':optimizer.state_dict()}
            torch.save(pretrain_state_dict, path_pretrain)
                
        # Training
        if pretrain:
            optim_stdict = torch.load(path_pretrain)['optimizer']
            model_stdict = torch.load(path_pretrain)['model']
            psi_ann.load_state_dict(model_stdict)
            optimizer.load_state_dict(optim_stdict)
            psi_ann.train()
        path_plot = f'saved_models/HO/{d}D/{arch}/nhid{nhid}/optim{optim}/' \
                    f'{actfun}/lr{lr}/alpha{alpha}/mu{mom}/plots/' \
                    f'seed{seed}_epochs{epochs}'
        # We store the energy data in lists for later plotting
        E_accum = []
        K_accum = []
        U_accum = []
        psi_accum = []
        overlap_accum = []
        
        for t in tqdm(range(epochs)):
            loss_fn = HO_energy_2der if nders == 2 else HO_energy
            E, K, U, psi, overlap = loss_fn(model=psi_ann, 
                                      train_data=mesh, 
                                      w_i=w_trpz, 
                                      x2_y2_z2=x2_y2_z2,
                                      target_dD=target_dD,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph)
            #show_layers(psi_ann)
            if t == 0:
                if computation_graph:
                    make_dot(E, params=dict(list(psi_ann.named_parameters())), 
                             show_attrs=show_attrs, show_saved=show_saved).render(cg_name, 
                                                                                  format='pdf')
                print(f'\Right after pretraining: \nOverlap: {overlap}, \nK: {K}, ' \
                      f'\nU: {U}, \nE: {E}')
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
            E_accum.append(E.item())
            K_accum.append(K.item())
            U_accum.append(U.item())
            psi_accum.append(psi.detach())
            overlap_accum.append(overlap.item())
            
            # Plotting
            if (((t+1) % leap) == 0 and periodic_plots) or t == epochs-1:
                minimisation_plots(x_axis=[i for i in range(t+1)], 
                                   y_axis=E_accum,
                                   K=K_accum,
                                   U=U_accum,
                                   y_exact=E_theoretical,
                                   d=d,
                                   x=x,
                                   y=x,
                                   z=psi, 
                                   overlap=overlap_accum,
                                   path_plot=path_plot,
                                   show_last_plot=show_last_plot,
                                   save=save_plot if t == epochs - 1 else False)
                
        # Console feedback
        print('\nModel trained!')
        print('Total execution time:  {:6.2f} seconds ' \
              '(run on {})'.format(time.time() - start_time_all, device))
                
        if save_model:
            path_model = f'saved_models/HO/{d}D/{arch}/nhid{nhid}/optim{optim}/' \
                f'{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/' \
                f'seed{seed}_epochs{epochs}'
            full_path_model = f'{path_model}.pt'
            state_dict = {'loss': E,
                          'epochs': epochs,
                          'model_state_dict':psi_ann.state_dict(),
                          'optimizer_state_dict':optimizer.state_dict()}
            torch.save(state_dict, full_path_model)
            print(f'Model saved in {full_path_model}')
        if save_plot: 
            full_path_plot = f'{path_plot}.pdf'
            print(f'Plot saved in {full_path_plot}')
            
print("\nAll done! :)")
