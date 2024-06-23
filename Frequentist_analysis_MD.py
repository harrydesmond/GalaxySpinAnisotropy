# Use isotropic mock data to assess the frequentist p-value of the dipole magnitude in each dataset, and create contour plots
# Parallelised MPI over mock dataset generation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from matplotlib.patches import PathPatch
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
import sys

data_choice = int(sys.argv[1])      # Which dataset to analyse
choice = int(sys.argv[2])           # Which repeat number to use

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.random.seed(rank)

randomise_spins = True
randomise_positions = False

if data_choice==0: filename = 'Longo.csv'
elif data_choice==1: filename = 'Iye.csv'
elif data_choice==2: filename = 'SDSS_DR7.csv'
elif data_choice==3: filename = 'GAN_M.csv'
elif data_choice==4: filename = 'GAN_NM.csv'
elif data_choice==5: filename = 'Shamir.csv'
elif data_choice==6: filename = 'PS_DR1.csv'
else:
    print("Wrong dataset choice")
    quit()

if rank==0: print("Analysing dataset", filename)

name = filename[:-4]

data = pd.read_csv("Datasets/"+filename)

if filename == 'Longo.csv':
    data_low_z = data[(data['RS'] < 0.04) & (data['G'] < 17)]
    if rank==0: print("Number of galaxies after G < 17 cut for z < 0.04:", len(data_low_z))

    data_high_z = data[(data['RS'] >= 0.04) & (data['G'] < 17.4)]
    if rank==0: print("Number of galaxies after G < 17.4 cut for z >= 0.04:", len(data_high_z))

    final_data = data[(data['G'] <= 17.4) & (data['RS'] <= 0.085)]
    if rank==0: print("Number of galaxies after G <= 17.4 and z <= 0.085 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) > 1.6]
    if rank==0: print("Number of galaxies after (U - Z) > 1.6 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) < 3.5]
    if rank==0: print("Number of galaxies after (U - Z) < 3.5 cut:", len(final_data))

    data = final_data

    data['Spin'] = data['Spin'].apply(lambda x: 1 if x == 'L' else 0)
    cw_ccw = data['Spin'].values
else:
    data['cw/ccw'] = data['cw/ccw'].apply(lambda x: 1 if x == 1 else 0)
    cw_ccw = data['cw/ccw'].values

RA = np.radians(data['RA'])
Dec = np.radians(data['Dec'])
n_ra = np.radians(data['RA'])
n_dec = np.radians(data['Dec'])
n_theta = np.pi/2 - n_dec

spin = np.array(cw_ccw)

def lnprob(x, *args):
    M = x[0]
    D, d_ra, d_dec = x[1], x[2], x[3]

    if M < 0.3 or M > 0.7 or D > 0.3 or D < 0.:
        return -np.inf

    if d_dec > np.pi/2 or d_dec < -np.pi/2 or d_ra > 2.*np.pi or d_ra < 0:
        return -np.inf

    d_theta = np.pi/2 - d_dec

    prob = M + D * (np.sin(d_theta) * np.sin(n_theta) * np.cos(d_ra - n_ra) + np.cos(d_theta) * np.cos(n_theta))

    if np.sum(prob < 0) > 0 or np.sum(prob > 1) > 0:
        if rank==0: print("Sampling P<0 or P>1:", M, D, flush=True)
        return -np.inf

    lnlike = np.sum(np.log(prob) * spin + np.log(1 - prob) * (1 - spin))
    lnprior = np.log(np.cos(d_dec))

    return lnlike + lnprior

def nll(x):
    return -lnprob(x)

min_lnlike_real = np.inf
best_params_real = []
for m in range(20):     # 40
    inpt = [np.random.uniform(0.4, 0.6), np.random.uniform(0., 0.2), np.random.uniform(0, 2.*np.pi), np.random.uniform(-np.pi/2, np.pi/2)]
    res = minimize(nll, inpt, method="Nelder-Mead")
    if res['fun'] < min_lnlike_real:
        best_params_real = res.x
        min_lnlike_real = res['fun']

if rank==0: print("Real Data MLE params:", best_params_real, min_lnlike_real)

M_true, D_true = best_params_real[0], 0
dra_true, ddec_true = np.pi, -np.pi/4

Nmock_tot = 50000

if Nmock_tot!=50000 and rank==0:
    print("As you are not using 50000 mock datasets, you may need to adjust the num_bins parameter for a nice-looking contour plot")

Nmock = int(Nmock_tot/size)

m_values, d_values = np.zeros(Nmock), np.zeros(Nmock)

if rank==0: mock_data_analysis_progress = tqdm(total=Nmock, desc='Mock Data Analysis')

for i in range(Nmock):
    if randomise_positions:
        n_ra, n_dec = np.random.uniform(0, 2*np.pi, len(spin)), np.arccos(np.random.uniform(-1, 1, len(spin))) - np.pi/2
    else:
        n_ra, n_dec = np.radians(data['RA']), np.radians(data['Dec'])

    n_theta = np.pi/2 - n_dec

    if randomise_spins:
        dtheta_true = np.pi/2 - ddec_true
        prob = M_true + D_true * (np.sin(dtheta_true) * np.sin(n_theta) * np.cos(dra_true - n_ra) + np.cos(dtheta_true) * np.cos(n_theta))
        rand = np.random.uniform(0, 1, len(n_ra))
        spin = np.zeros(len(n_ra))
        spin[prob > rand] = 1
    else:
        spin = np.array(cw_ccw)

    min_lnlike_mock = np.inf

    for m in range(40):
        inpt = [np.random.uniform(0.4, 0.6), np.random.uniform(0., 0.2), np.random.uniform(0, 2.*np.pi), np.random.uniform(-np.pi/2, np.pi/2)]
        res = minimize(nll, inpt, method="Nelder-Mead")
        if res['fun'] < min_lnlike_mock:
            best_params_mock = res.x
            min_lnlike_mock = res['fun']

    m_values[i], d_values[i] = best_params_mock[0], best_params_mock[1]
    
    if rank==0: mock_data_analysis_progress.update(1)

if rank==0: mock_data_analysis_progress.close()

buff = np.array(list(m_values)+list(d_values))

assert len(buff)==Nmock*2

if rank==0:            # This and the below are only done for a single thread. Everything has to finish.
    final_arr = np.zeros([size, Nmock*2])
    final_arr[0] = buff
    for i in range(1, size):
        comm.Recv(buff, source=i)        # Just to make sure everything's finished before you do the final step below
        final_arr[i]=buff
else:
    comm.Send(buff, dest=0)

if rank==0:
    print("Starting to make contour plot", flush=True)

    x = np.ndarray.flatten(final_arr[:,:Nmock])         # Combined over all procs; M values
    y = np.ndarray.flatten(final_arr[:,Nmock:])         # D values

    np.save('m_values_'+str(name)+'_'+str(choice)+'.npy', x)
    np.save('d_values_'+str(name)+'_'+str(choice)+'.npy', y)

    assert len(x)==Nmock*size

    total_datasets_mock = len(x)

    if data_choice==0:
        num_bins = 32
        y_max = 0.035
    elif data_choice==1:
        num_bins = 40
        y_max = 0.015
    elif data_choice==2:
        num_bins = 28
        y_max = 0.15
    elif data_choice==3:
        num_bins = 26
        y_max = 0.011
    elif data_choice==4:
        num_bins = 32
        y_max = 0.011
    elif data_choice==5:
        num_bins = 40
        y_max = 0.015
    else:
        num_bins = 32
        y_max = 0.023

    x_bins = np.linspace(min(x), max(x), num_bins + 1)
    y_bins = np.linspace(min(y), max(y), num_bins + 1)

    H_mock, x_edges_mock, y_edges_mock = np.histogram2d(x, y, bins=(x_bins, y_bins))

    contour = plt.contourf(x_edges_mock[:-1], y_edges_mock[:-1], H_mock.T / total_datasets_mock, levels=30, cmap='viridis')

    colorbar = plt.colorbar(contour, label='', orientation='vertical', fraction=0.05, pad=0.001)

    colorbar.ax.tick_params(labelsize=10)

    desired_fractions = [0.683, 0.954]

    total_count_mock = np.sum(H_mock)

    threshold_counts = [fraction * total_count_mock for fraction in desired_fractions]

    sorted_histogram_values = np.sort(H_mock.flatten())[::-1]

    cumulative_counts = np.cumsum(sorted_histogram_values)

    bin_indices = [np.where(cumulative_counts >= threshold_count)[0][0] for threshold_count in threshold_counts]

    threshold_values = sorted_histogram_values[bin_indices]

    contour_levels = sorted(threshold_values)

    Hmock_interp = RegularGridInterpolator((x_edges_mock[:-1], y_edges_mock[:-1]), H_mock, method="cubic")
    clev_data = Hmock_interp((best_params_real[0], best_params_real[1]))

    frac_lev = np.sum(H_mock.flatten()[H_mock.flatten() >= clev_data])/total_count_mock
    p_value = round(1-frac_lev,3)

    contour_levels = sorted(list(contour_levels) + [clev_data])

    if data_choice != 2:
        contour_lines_mock = plt.contour(x_edges_mock[:-1], y_edges_mock[:-1], H_mock.T, levels=contour_levels, colors='white', linestyles='solid', linewidths=1.5)
        sigma = 0.80
        for collection in contour_lines_mock.collections:
            paths = collection.get_paths()
            for path in paths:
                path.vertices[:, 0] = gaussian_filter(path.vertices[:, 0], sigma)
                path.vertices[:, 1] = gaussian_filter(path.vertices[:, 1], sigma)

    plt.ylim(0, y_max)

    new_des_frac = sorted(np.array(list(desired_fractions) + [frac_lev]), reverse=True)

    if data_choice != 2:
        plt.clabel(contour_lines_mock, inline=True, fontsize=7, fmt={level: f'{fraction*100:.1f}%' for level, fraction in zip(contour_lines_mock.levels, new_des_frac)}, inline_spacing=10, rightside_up=True, manual=False, zorder=3)

    scatter_point = plt.scatter(best_params_real[0], best_params_real[1], c='red', marker='+', label='Observed MLE', s=35, zorder=4, linewidths=1.5)

    plt.legend(handles=[scatter_point], labels = ['p = {:.3f}'.format(p_value)], loc='upper right', frameon=False, labelcolor='white', fontsize=11)

    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.xlabel(r'$\widehat{M}$', fontsize=12, labelpad=7.0)
    plt.ylabel(r'$\widehat{D}$', fontsize=12, labelpad=7.0)

    plt.grid(False, which='both')

    plt.savefig("Plots/"+name+"_frequentist_Final"+str(choice)+".png", bbox_inches="tight")

    print("p value:", p_value)
