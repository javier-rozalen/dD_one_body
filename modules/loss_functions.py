# -*- coding: utf-8 -*-
######### IMPORTS ##########
import torch

############################ LOSS FUNCTIONS ##############################
def HO_energy(model, train_data, w_i, x2_y2_z2, target_dD):
    
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
    
    # Overlap
    overlap = torch.dot(psi_normalized * target_dD, w_i)
    #print(psi, d_psi, d_psi2, E)
    
    return E, K, U, psi_normalized, overlap**2

    