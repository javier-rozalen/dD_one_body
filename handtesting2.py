#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:43:03 2023

@author: jozalen
"""

import torch 

################### EPOCH 0 ###################
print('\nEPOCH 0\n--------------------------------------------------')
# Static params
mesh = torch.tensor([[-5., -5., -5.],
        [-5., -5.,  5.],
        [-5.,  5., -5.],
        [-5.,  5.,  5.],
        [ 5., -5., -5.],
        [ 5., -5.,  5.],
        [ 5.,  5., -5.],
        [ 5.,  5.,  5.]])
W1 = torch.tensor([[-0.7576, -0.2793, -0.4031],
        [-0.7347, -0.0293, -0.7999]])
W111 = W1[0][0]
W112 = W1[0][1]
W113 = W1[0][2]
W121 = W1[1][0]
W122 = W1[1][1]
W123 = W1[1][2]
B1 = torch.tensor([-0.2057,  0.5087])
W2 = torch.tensor([[0.5695, 0.4388]])
W21 = W2[0][0]
W22 = W2[0][1]
h = 5.
d = 3
sigmoid = torch.nn.Sigmoid()
gamma = 0.05 # lr
eps = 1e-8
alpha = 0.9
v = 0.

############################ W111 ############################
# Functions of x,y,z
r = lambda x, y, z : torch.tensor([x, y, z])
in1 = lambda x, y, z : torch.dot(W1[0],r(x,y,z)) + B1[0]
in2 = lambda x, y, z : torch.dot(W1[1],r(x,y,z)) + B1[1]
dsigm_din1 = lambda x, y, z : torch.exp(-in1(x,y,z)) \
    /(1+torch.exp(-in1(x,y,z))).pow(2)
dsigm_din2 = lambda x, y, z : torch.exp(-in2(x,y,z)) \
    /(1+torch.exp(-in2(x,y,z))).pow(2)
ddsigmdin1_dW111 = lambda x, y, z : x*torch.exp(-in1(x,y,z))*(torch.exp(-in1(x,y,z))-1) \
    /(1+torch.exp(-in1(x,y,z))).pow(3)
psi = lambda x, y, z : W21*sigmoid(in1(x,y,z))+W22*sigmoid(in2(x,y,z))
dpsi_dx = lambda x, y, z : W21*dsigm_din1(x,y,z)*W111+W22*dsigm_din2(x,y,z)*W121
dpsi_dy = lambda x, y, z : W21*dsigm_din1(x,y,z)*W112+W22*dsigm_din2(x,y,z)*W122
dpsi_dz = lambda x, y, z : W21*dsigm_din1(x,y,z)*W113+W22*dsigm_din2(x,y,z)*W123
dpsi_dW111 = lambda x, y, z : W21*dsigm_din1(x,y,z)*x
ddpsidx_dW111 = lambda x, y, z : W21*(ddsigmdin1_dW111(x,y,z)*W111+dsigm_din1(x,y,z))
ddpsidy_dW111 = lambda x, y, z : W21*W112*ddsigmdin1_dW111(x,y,z)
ddpsidz_dW111 = lambda x, y, z : W21*W113*ddsigmdin1_dW111(x,y,z)

# Sums over x,y,z
N = 0.
for x, y, z in mesh:
    N += psi(x,y,z)**2
N *= (h**d)

dN_dW111 = 0.
for x, y, z in mesh:
    dN_dW111 += psi(x,y,z)*dpsi_dW111(x,y,z)
dN_dW111 *= 2*(h**d)

num_dU_dW111 = 0.
for x, y, z in mesh:
    num_dU_dW111 += (x**2+y**2+z**2)*(2*N*psi(x,y,z)*dpsi_dW111(x,y,z)-((psi(x,y,z))**2)*dN_dW111)
dU_dW111 = (h**d) * num_dU_dW111 / (2*N**2)

num_dK_dW111 = 0.
for x, y, z in mesh:
    num_dK_dW111 += 2*N*(ddpsidx_dW111(x,y,z)*dpsi_dx(x,y,z)+ddpsidy_dW111(x,y,z)*dpsi_dy(x,y,z)+ddpsidz_dW111(x,y,z)*dpsi_dz(x,y,z)) - \
        dN_dW111*(dpsi_dx(x,y,z)**2+dpsi_dy(x,y,z)**2+dpsi_dz(x,y,z)**2)
dK_dW111 = (h**d) * num_dK_dW111 / (2*N**2)

dE_dW111 = dK_dW111 + dU_dW111

W111_after = W111 - gamma * dE_dW111

U = 0.
for x, y, z in mesh:
    U += (x**2+y**2+z**2)*((W21*sigmoid(in1(x,y,z))+W22*sigmoid(in2(x,y,z))).pow(2))
U *= (h**d)/(2*N)    
    
print(f'U: {U}\n')
print(f'W111: {W111}, \nW111_after: {W111_after}\ndE_dW111: {dE_dW111}\n')

############################ W112 ############################
# Functions of x,y,z
r = lambda x, y, z : torch.tensor([x, y, z])
in1 = lambda x, y, z : torch.dot(W1[0],r(x,y,z)) + B1[0]
in2 = lambda x, y, z : torch.dot(W1[1],r(x,y,z)) + B1[1]
dsigm_din1 = lambda x, y, z : torch.exp(-in1(x,y,z)) \
    /(1+torch.exp(-in1(x,y,z))).pow(2)
dsigm_din2 = lambda x, y, z : torch.exp(-in2(x,y,z)) \
    /(1+torch.exp(-in2(x,y,z))).pow(2)
ddsigmdin1_dW112 = lambda x, y, z : y*torch.exp(-in1(x,y,z))*(torch.exp(-in1(x,y,z))-1) \
    /(1+torch.exp(-in1(x,y,z))).pow(3)
psi = lambda x, y, z : W21*sigmoid(in1(x,y,z))+W22*sigmoid(in2(x,y,z))
dpsi_dx = lambda x, y, z : W21*dsigm_din1(x,y,z)*W111+W22*dsigm_din2(x,y,z)*W121
dpsi_dy = lambda x, y, z : W21*dsigm_din1(x,y,z)*W112+W22*dsigm_din2(x,y,z)*W122
dpsi_dz = lambda x, y, z : W21*dsigm_din1(x,y,z)*W113+W22*dsigm_din2(x,y,z)*W123
dpsi_dW112 = lambda x, y, z : W21*dsigm_din1(x,y,z)*y
ddpsidx_dW112 = lambda x, y, z : W21*W111*ddsigmdin1_dW112(x,y,z)
ddpsidy_dW112 = lambda x, y, z : W21*(ddsigmdin1_dW112(x,y,z)*W112+dsigm_din1(x,y,z))
ddpsidz_dW112 = lambda x, y, z : W21*W113*ddsigmdin1_dW112(x,y,z)

# Sums over x,y,z
N = 0.
for x, y, z in mesh:
    N += psi(x,y,z)**2
N *= (h**d)

dN_dW112 = 0.
for x, y, z in mesh:
    dN_dW112 += psi(x,y,z)*dpsi_dW112(x,y,z)
dN_dW112 *= 2*(h**d)

num_dU_dW112 = 0.
for x, y, z in mesh:
    num_dU_dW112 += (x**2+y**2+z**2)*(2*N*psi(x,y,z)*dpsi_dW112(x,y,z)-((psi(x,y,z))**2)*dN_dW112)
dU_dW112 = (h**d) * num_dU_dW112 / N**2

num_dK_dW112 = 0.
for x, y, z in mesh:
    num_dK_dW112 += 2*N*(ddpsidx_dW112(x,y,z)*dpsi_dx(x,y,z)+ddpsidy_dW112(x,y,z)*dpsi_dy(x,y,z)+ddpsidz_dW112(x,y,z)*dpsi_dz(x,y,z)) - \
        dN_dW112*(dpsi_dx(x,y,z)**2+dpsi_dy(x,y,z)**2+dpsi_dz(x,y,z)**2)
dK_dW112 = (h**d) * num_dK_dW112 / (2*N**2)

dE_dW112 = dK_dW112 + dU_dW112

W112_after = W112 - gamma * dE_dW112  
    
print(f'W112: {W112}, \nW112_after: {W112_after}\ndE_dW112: {dE_dW112}\n')


