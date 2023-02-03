# -*- coding: utf-8 -*-
######### IMPORTS ##########
import torch

############################ LOSS FUNCTIONS ##############################
def HO_energy(model, train_data, x2_y2_z2):
    # Psi
    psi = model(train_data) # wave function, dimension = (N)    
    # First derivative
    v = torch.ones_like(psi)
    d_psi, = torch.autograd.grad(outputs=psi, 
                                 inputs=train_data, 
                                 grad_outputs=v,
                                 retain_graph=True,
                                 create_graph=True)
    # Second derivative
    d2_psi, = torch.autograd.grad(outputs=d_psi,
                                  inputs=train_data,
                                  grad_outputs=v,
                                  retain_graph=True,
                                  create_graph=False)
    
    # Local energy computation
    laplacian_vec = torch.sum(d2_psi, dim=1) # Local kinetic energy
    potential_vec = x2_y2_z2 # Local "potential energy"
    E_L_vec = potential_vec - (laplacian_vec / psi) # Local energy
    E = 0.5 * torch.sum(E_L_vec) # "energy"
    print(psi, psi.shape, '\n', d_psi, d_psi.shape, '\n', d2_psi, d2_psi.shape)
    
    return E, psi

    