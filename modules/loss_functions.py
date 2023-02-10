# -*- coding: utf-8 -*-
######### IMPORTS ##########
import torch

############################ LOSS FUNCTIONS ##############################
def HO_energy(model, train_data, w_i, x2_y2_z2, target_dD, retain_graph,
              create_graph):
    # phi, d_phi, norm
    psi = model(train_data) # wave function, dimension = (N)    
    psi2 = psi ** 2
    d_psi, = torch.autograd.grad(outputs=psi, 
                                 inputs=train_data, 
                                 grad_outputs=torch.ones_like(psi),
                                 retain_graph=retain_graph,
                                 create_graph=create_graph)
    d_psi2 = d_psi **2
    N = w_i * torch.sum(psi2) # squared ANN norm
    psi_normalized = psi / torch.sqrt(N)
    
    # ANN energy
    K_vec = torch.sum(d_psi2, dim=1) # Local K
    U_vec = (psi2) * x2_y2_z2 # Local U
    K = 0.5 * w_i * torch.sum(K_vec) / N # Integrated K
    U = 0.5 * w_i * torch.sum(U_vec) / N # Integrated U
    E = K + U 
    
    # Overlap
    overlap = w_i * torch.sum(psi_normalized * target_dD)
    """
    print(f'\nmesh: {train_data}, {train_data.shape}, \npsi: {psi}, {psi.shape},' \
          f'\nd_psi: {d_psi}, {d_psi.shape}, \nK: {K}, \nU: {U}, \nE: {E}')
    """
    return E, K, U, psi_normalized, overlap**2

def HO_energy_2der(model, train_data, w_i, x2_y2_z2, target_dD, retain_graph,
              create_graph):
    # phi, d_phi, norm
    psi = model(train_data) # wave function, dimension = (N)    
    psi2 = psi ** 2
    d_psi, = torch.autograd.grad(outputs=psi, 
                                 inputs=train_data, 
                                 grad_outputs=torch.ones_like(psi),
                                 retain_graph=retain_graph,
                                 create_graph=create_graph)
    d2_psi, = torch.autograd.grad(outputs=d_psi,
                                  inputs=train_data,
                                  grad_outputs=torch.ones_like(d_psi),
                                  retain_graph=retain_graph,
                                  create_graph=create_graph)
    N = w_i * torch.sum(psi2) # squared ANN norm
    psi_normalized = psi / torch.sqrt(N)
    
    # ANN energy
    K_vec = -torch.sum(d2_psi, dim=1) # Local K
    U_vec = (psi2) * x2_y2_z2 # Local U
    K = 0.5 * w_i * torch.sum(K_vec) / N # Integrated K
    U = 0.5 * w_i * torch.sum(U_vec) / N # Integrated U
    E = K + U 
    
    # Overlap
    overlap = w_i * torch.sum(psi_normalized * target_dD)
    
    print(f'\nmesh: {train_data}, {train_data.shape}, \npsi: {psi}, {psi.shape},' \
          f'\nd_psi: {d_psi}, {d_psi.shape}, \nd2_psi: {d2_psi}, {d2_psi.shape},' \
             f', \nK: {K}, \nU: {U}, \nE: {E}')
    
    return E, K, U, psi_normalized, overlap**2

def overlap(model, train_data, w_i, target_dD):
    # Psi
    psi = model(train_data) # wave function, dimension = (N)    
    N = w_i * torch.sum(psi**2) # squared ANN norm
    psi_normalized = psi / torch.sqrt(N)
    overlap = w_i * torch.sum(psi_normalized * target_dD)
    
    return overlap, (1-overlap)**2, psi_normalized

def MSE(model, train_data, w_i, target_dD):
    # Psi
    psi = model(train_data) # wave function, dimension = (N)    
    N = w_i * torch.sum(psi**2) # squared ANN norm
    psi_normalized = psi / torch.sqrt(N)
    mse = torch.nn.MSELoss()(psi_normalized, target_dD)
    
    return mse, psi_normalized


    