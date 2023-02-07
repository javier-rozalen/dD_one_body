# -*- coding: utf-8 -*-
######### IMPORTS ##########
import torch

############################ LOSS FUNCTIONS ##############################
def HO_energy(model, train_data, w_i, x2_y2_z2):
    # Psi
    psi = model(train_data) # wave function, dimension = (N)    
    # First derivative
    d_psi, = torch.autograd.grad(outputs=psi, 
                                 inputs=train_data, 
                                 grad_outputs=torch.ones_like(psi),
                                 retain_graph=True,
                                 create_graph=True)
    # Second derivative
    d2_psi, = torch.autograd.grad(outputs=d_psi,
                                  inputs=train_data,
                                  grad_outputs=torch.ones_like(d_psi),
                                  retain_graph=True,
                                  create_graph=False)
    
    # Local energy computation
    laplacian_vec = torch.sum(d2_psi, dim=1) # Local kinetic energy
    potential_vec = x2_y2_z2 # Local "potential energy"
    E_L_vec = potential_vec - (laplacian_vec / psi) # Local energy
    E = 0.5 * torch.sum(E_L_vec) / train_data.shape[0] # "energy"
    #print(psi, psi.shape, '\n', d_psi, d_psi.shape, '\n', d2_psi, d2_psi.shape)
    
    return E, psi / torch.sqrt(torch.dot(psi**2, w_i))

def HO_energy_trick(model, train_data, w_i, x2_y2_z2):
    # phi, d_phi, norm
    psi = model(train_data) # wave function, dimension = (N)    
    psi2 = psi ** 2
    d_psi, = torch.autograd.grad(outputs=psi, 
                                 inputs=train_data, 
                                 grad_outputs=torch.ones_like(psi),
                                 retain_graph=True)
    d_psi2 = d_psi **2
    N = torch.dot(psi2, w_i) # squared ANN norm
    psi_normalized = psi / torch.sqrt(N)
    
    # ANN energy
    K_vec = torch.sum(d_psi2, dim=1) # Local K
    U_vec = (psi2) * x2_y2_z2 # Local U
    K = 0.5 * torch.dot(K_vec, w_i) / N # Integrated K
    U = 0.5 * torch.dot(U_vec, w_i) / N # Integrated U
    E = K + U 
    
    return E, psi_normalized

def overlap(model, target, w_i, train_data):
    # Psi
    psi = model(train_data) # wave function, dimension = (N)    
    psi_normalized = psi / torch.sqrt(torch.dot(psi**2, w_i))
    
    overlap = torch.dot(psi_normalized * target, w_i)
    
    return (1-overlap)**2, psi_normalized



    