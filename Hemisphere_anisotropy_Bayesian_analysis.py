# Calculate parameter constraints and create corner plot in a Bayesian MCMC analysis
# This code infers monopole, dipole and quadrupole, but can easily be modified to restrict to some subset
# Parallelised Open-MP in MCMC sampling

import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt
import corner
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import sys

data_choice = int(sys.argv[1])

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

print("Analysing dataset", filename)

name = filename[:-4]

data = pd.read_csv("Datasets/"+filename)

if filename == 'Longo.csv':
    data_low_z = data[(data['RS'] < 0.04) & (data['G'] < 17)]
    print("Number of galaxies after G < 17 cut for z < 0.04:", len(data_low_z))

    data_high_z = data[(data['RS'] >= 0.04) & (data['G'] < 17.4)]
    print("Number of galaxies after G < 17.4 cut for z >= 0.04:", len(data_high_z))

    final_data = data[(data['G'] <= 17.4) & (data['RS'] <= 0.085)]
    print("Number of galaxies after G <= 17.4 and z <= 0.085 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) > 1.6]
    print("Number of galaxies after (U - Z) > 1.6 cut:", len(final_data))

    final_data = final_data[(final_data['U'] - final_data['Z']) < 3.5]
    print("Number of galaxies after (U - Z) < 3.5 cut:", len(final_data))

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

N_CW = np.sum(spin)
N_tot = len(spin)

print("Mean spin:", np.mean(spin))

def lnlike(x,*args):

    M = x[0]
    A, a_ra, a_dec = x[1], x[2], x[3]

    if a_dec > np.pi / 2 or a_dec < -np.pi / 2 or a_ra > 2. * np.pi or a_ra < 0:
        return -np.inf

    if M < 0.3 or M > 0.7 or A > 0.3 or A < 0.:
        return -np.inf

    cos_theta = (np.sin(a_dec) * np.sin(n_dec) + np.cos(a_dec) * np.cos(n_dec) * np.cos(a_ra - n_ra))

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    prob = np.where(theta < np.pi / 2, M + A, M - A)

    if np.any(prob < 0) or np.any(prob > 1):
        print("This shouldn't be happening", flush=True)
        return -np.inf

    lnlike = np.sum(np.log(prob) * spin + np.log(1 - prob) * (1 - spin))

    return lnlike

def lnprob(x,*args):

    a_dec = x[3]

    if a_dec>np.pi/2 or a_dec<-np.pi/2:
        return -np.inf
    
    lnprior = np.log(np.cos(a_dec))

    lnlike_value = lnlike(x,*args)
    
    return lnlike_value + lnprior

def nll(x):
    return -lnlike(x)

min_values = []
min1 = np.inf

for m in range(10):
    inpt = [np.random.uniform(0.4, 0.6), np.random.uniform(0., 0.2), np.random.uniform(0, 2.*np.pi), np.random.uniform(-np.pi/2, np.pi/2)
    res = minimize(nll, inpt, method="Nelder-Mead")
    if res['fun'] < min1:
        params = res.x
        min1 = res['fun']
    min_values.append("Iteration {} - Min value: {}".format(m+1, min1))

print("ML params:", params, min1)
# print("Min values:")
# for value in min_values:
    # print(value)

L = lnlike(params)

k = len(params)
n = len(data)

bic = -2 * (L) + k * np.log(n)

print("BIC:", bic)

bic_values_for_M = [
    21012.4110286252, # Longo.csv
    101051.90731787,  # Iye.csv
    8469.24333237917, # SDSS_DR7.csv
    193871.969825225, # GAN_M.csv
    192618.331159577, # GAN_NM.csv
    107916.724775013, # Shamir.csv
    39828.6986758339  # PS_DR1.csv
]

delta_bic = bic - bic_values_for_M[data_choice]

print("Delta BIC:", delta_bic)

ndim = 9
nwalkers = 22
max_iters = 600000

p0 = []     # Initial positions for the walkers
for i in range(nwalkers):
    M_init = np.random.uniform(0.3, 0.6)
    A_init = np.random.uniform(0, 0.2)
    a_ra_init = np.random.uniform(0, 2 * np.pi)
    a_dec_init = np.random.uniform(-np.pi / 2, np.pi / 2)
    
    pi = [M_init, A_init, a_ra_init, a_dec_init]
    p0.append(pi)

autocorr = np.empty(max_iters)
old_tau = np.inf

print("Running MCMC with", cpu_count(), "cores. Please wait...")

with Pool() as pool:
    filename = "chain_"+name+"_MDQ.h5"
    backend = emcee.backends.HDFBackend(filename)

    backend.reset(nwalkers, ndim)  # Restart the backend

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)

    for sample in sampler.sample(p0, iterations=max_iters, progress=True):
        if sampler.iteration == max_iters - 1:
            print("The sampler did not converge. Exiting.")
            break

        if sampler.iteration % 100 == 0:
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[sampler.iteration//100] = np.mean(tau)

            converged = np.all(tau * 100 < sampler.iteration)               # More stringent
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            # converged = np.all(tau * 50 < sampler.iteration)              # Less stringent
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.02)
            if converged:
                print("Convergence achieved. Exiting.")
                break
            old_tau = tau

print("Completed. Mean acceptance fraction =", round(np.mean(sampler.acceptance_fraction), 3))

reader = emcee.backends.HDFBackend(filename)

try:
    tau = reader.get_autocorr_time(tol=50, quiet=True)
    if np.isnan(tau).any():
        print("Tau contains nan", tau)
        raise emcee.autocorr.AutocorrError("Tau contains nan")
    
    burnin = int(4 * np.max(tau))
    thin = 1
    
    print("burn-in:", burnin)
    print("thin:", thin)

    if thin < 1:
        thin = 1

except emcee.autocorr.AutocorrError:
    print("Tau is not well defined")
    thin, burnin = 1, 400
    print("burnin: ", burnin)
    print("thin: ", thin)

lnlike = -1.e100

samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin)
log_prob_samples_flat = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

mask_like = log_prob_samples_flat > lnlike

samples = samples[mask_like,:]
log_prob_samples_flat = log_prob_samples_flat[mask_like]

nsteps = len(log_prob_samples[:,0])
ndim = len(samples[0,:])
nwalkers = len(log_prob_samples[0,:])
nsteps_mask = int(len(samples[:,0])/nwalkers)

print("nsteps:", nsteps, ", nsteps_mask:", nsteps_mask, ", ndim:", ndim, ", nwalkers:", nwalkers)

x = list(range(1, nsteps + 1))
for i in range(nwalkers):
    like_i = log_prob_samples[:,i]              
    mask = like_i > lnlike
    indices = np.where(mask)[0]
    plt.scatter(np.array(x)[indices] + burnin, like_i[indices], s=10)
plt.xlabel("Walker step")
plt.ylabel("ln(Likelihood)")
plt.savefig("Plots/"+name+"_MDQ_walkers.png", bbox_inches="tight")

labels = [r'${\rm \widehat{M}}$', r'${\rm \widehat{A}}$', r'${\rm a_{\alpha}}$', r'${\rm a_{\delta}}$']

plt.clf()

truth_values = [None] * ndim
truth_values[labels.index(r'${\rm \widehat{M}}$')] = 0.5

corner.corner(samples, labels=labels, show_titles=True, quantiles=[], plot_datapoints=False,
              title_kwargs={"fontsize": 18}, title_fmt='.3f', label_kwargs={"fontsize": 18},
              verbose=True, truths=truth_values)

plt.savefig("Plots/"+name+"_MDQ_corner.png", bbox_inches="tight")

param_means = np.mean(samples, axis=0)
param_std = np.std(samples, axis=0)

param_quotes = []

for i, (mean, std) in enumerate(zip(param_means, param_std)):
    if mean - 2 * std < 0:
        p68 = np.percentile(samples[:, i], 68)
        p95 = np.percentile(samples[:, i], 95)
        param_quotes.append(f'<{p68:.3f} at 1 sigma, <{p95:.3f} at 2 sigma')
    else:
        param_quotes.append("use value given by corner plot")

for i, param_quote in enumerate(param_quotes):
    print(f'Parameter {i+1}: {param_quote}')