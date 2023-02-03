# -*- coding: utf-8 -*-
######################## IMPORTS ########################
import torch, gc
import matplotlib.pyplot as plt
import numpy as np

######################## MISC. ########################
plt.rcParams['agg.path.chunksize'] = 10000

######################## PLOTTER FUNCTIONS ########################
def minimisation_plots(x_axis, y_axis, K, U, y_exact, d, x=[], y=[], z=[], 
                       overlap=[], path_plot='', show_last_plot=True, 
                       save=False):
    
    if d == 1:
        fig = plt.figure(figsize=(10, 8))
        
        # Wave function plot
        ax = fig.add_subplot(2, 2, 1)
        target = lambda x : (1/np.pi)**(1/4) * np.exp(-(x**2)/2)
        ax.plot(x.detach().numpy(), target(x.detach().numpy()), label='$\psi_{\mathrm{targ}}$', 
                   color='red', linestyle='dashed')
        ax.plot(x.detach().numpy(), z.detach().numpy(), color='blue', label='$\psi_{\mathrm{ANN}}$')
        ax.legend(fontsize=12)
        
        # Energies plot 
        ax = fig.add_subplot(2, 1, 2)
        ax.set_ylim(-1, 5)
        ax.plot(x_axis, y_axis, label='E', color='blue')
        ax.plot(x_axis, K, label='K', color='orange')
        ax.plot(x_axis, U, label='U', color='green')
        ax.axhline(y_exact, color='blue', linestyle='dashed', label='E_th')
        ax.axhline(y_exact/2, color='orange', linestyle='dashed', label='K_th, U_th')
        ax.legend(fontsize=12)
        
        # Overlap plot
        ax = fig.add_subplot(2, 2, 2)
        ax.set_ylim(0, 1.05)
        ax.plot(x_axis, overlap, color='blue', label='Overlap')
        ax.axhline(1., color='blue', linestyle='dashed')
        ax.legend(fontsize=12)
        
        
    elif d == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Wave function plot
        sqrt_N = int(np.sqrt(len(z)))
        z = z.reshape(sqrt_N, sqrt_N).detach().numpy()
        x, y = np.meshgrid(x.detach().numpy(), y.detach().numpy())
        ax.plot_wireframe(x, y, z, linewidth=0.7, label='$\psi_{\mathrm{ANN}}$', color='blue')
        target = lambda x, y : (1/np.pi)**(1/2) * np.exp(-(x**2+y**2)/2)
        ax.plot_wireframe(x, y, target(x, y), color='green', linewidth=0.7,
                          label='$\psi_{\mathrm{targ}}$', linestyle='dashed')
        
        # Legend
        ax.legend(fontsize=12)
        
        # Energies plot
        ax = fig.add_subplot(2, 1, 2)
        ax.set_ylim(-1, 5)
        ax.plot(x_axis, y_axis)
        ax.plot(x_axis, K, label='K', color='orange')
        ax.plot(x_axis, U, label='U', color='green')
        ax.axhline(y_exact, color='blue', linestyle='dashed', label='E_th')
        ax.axhline(y_exact/2, color='orange', linestyle='dashed', label='K_th, U_th')
        
        # Legend
        ax.legend(fontsize=12)
        
        # Overlap plot
        ax = fig.add_subplot(2, 2, 2)
        ax.set_ylim(0, 1.05)
        ax.plot(x_axis, overlap, color='blue', label='Overlap')
        ax.axhline(1., color='blue', linestyle='dashed')
        
        # Legend
        ax.legend(fontsize=12)
        
    else:
        fig = plt.figure(figsize=(10, 8))
        
        # Energies plot
        ax = fig.add_subplot(1, 2, 1)
        ax.set_ylim(-1, 5)
        ax.plot(x_axis, y_axis, label='E', color='blue')
        ax.plot(x_axis, K, label='K', color='orange')
        ax.plot(x_axis, U, label='U', color='green')
        ax.axhline(y_exact, color='blue', linestyle='dashed', label='E_th') 
        ax.axhline(y_exact/2, color='orange', linestyle='dashed', label='K_th, U_th')
        
        # Legend
        ax.legend(fontsize=12)
        
        # Overlap plot
        ax = fig.add_subplot(1, 2, 2)
        ax.set_ylim(0, 1.05)
        ax.plot(x_axis, overlap, color='blue', label='Overlap')
        ax.axhline(1., color='blue', linestyle='dashed')
        
        # Legend
        ax.legend(fontsize=12)
        
    if save:
        full_path_plot = f'{path_plot}.pdf'
        plt.savefig(full_path_plot)
    
    if not show_last_plot:
        plt.clf()
        plt.close('all')
        gc.collect()
        
    plt.show()
