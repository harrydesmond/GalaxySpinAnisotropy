# Investigate biases in mock data given injection of arbitrary monopole and dipole
# Parallelised MPI over mock data generation
# Uses numpyro (built on jax) for more efficient sampling

import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random, lax, local_device_count
import matplotlib.pyplot as plt
import corner
import jax.numpy as jnp
import sys
from numpyro.distributions.util import promote_shapes
from scipy.optimize import minimize
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")
import math
from scipy.special import erf
import gc
from numpyro.diagnostics import effective_sample_size
from jax.tree_util import tree_flatten
from jax import device_get
from scipy.optimize import least_squares
from scipy.special import erf
import seaborn as sns

def equations(p):
    x, y = p            # x is mu, y is sigma
    return (x+(np.exp(-(x**2/(2*y**2)))*np.sqrt(2/math.pi)*y)/(1+erf(x/(np.sqrt(2)*y)))-tg_mean, y*(y-(2*np.exp(-(x**2/y**2))*y)/(math.pi*(1+erf(x/(np.sqrt(2)*y)))**2)+(np.exp(-(x**2/(2*y**2)))*np.sqrt(2/math.pi)*x)/(-2+(1-erf(x/(np.sqrt(2)*y)))))-tg_var)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel = False

#randomise_spins = bool(int(sys.argv[1]))
#randomise_positions = bool(int(sys.argv[2]))
randomise_spins = True
randomise_positions = False

allow_neg_D = False

if randomise_spins and rank==0:
    print("Randomising spins")

if randomise_positions and rank==0:
    print("Randomising positions")

if not randomise_spins and not randomise_positions and rank==0:
    print("You should be randomising something in this script")
    quit()

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

if parallel:
    import multiprocessing
    numpyro.set_host_device_count(multiprocessing.cpu_count())
    if rank==0:
        print("Parallelised with:", local_device_count(), multiprocessing.cpu_count(), flush=True)
else:
    if rank==0:
        print("Parallelisation disabled", flush=True)

progress = False


class mylike(dist.Distribution):
   
    def __init__(self, M, D, d_ra, d_dec):
        self.M, self.D, self.d_ra, self.d_dec = promote_shapes(M, D, d_ra, d_dec)
        super(mylike, self).__init__(batch_shape = ())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        M, D, d_ra, d_dec = self.M, self.D, self.d_ra, self.d_dec

        d_theta = jnp.pi/2 - d_dec

        prob = M + D*(jnp.sin(d_theta)*jnp.sin(n_theta)*jnp.cos(d_ra-n_ra) + jnp.cos(d_theta)*jnp.cos(n_theta))

        lnlike = jnp.sum(jnp.log(prob)*spin + jnp.log(1-prob)*(1-spin))

        lnprior = jnp.log(jnp.cos(d_dec))
        
        return lnlike + lnprior

    
def model():
    M = numpyro.sample("M", dist.Uniform(0.3, 0.7))
    if allow_neg_D:
        D = numpyro.sample("D", dist.Uniform(-0.3, 0.3))
        d_ra = numpyro.sample("d_ra", dist.Uniform(0, jnp.pi))
    else:    
        D = numpyro.sample("D", dist.Uniform(0, 0.3))
        d_ra = numpyro.sample("d_ra", dist.Uniform(0, 2*jnp.pi))
    d_dec = numpyro.sample("d_dec", dist.Uniform(-jnp.pi/2, jnp.pi/2))

    numpyro.sample('obs', mylike(M, D, d_ra, d_dec), obs=())


# ------------

nwarm, nsamp = 1000, 6000

M_true, D_true = 0.6, 0.2                   # Choose a monopole and dipole to inject
dra_true, ddec_true = np.pi, -np.pi/4

data = np.genfromtxt("Iye.csv", delimiter=",")     #ID, RA, Dec, cw/ccw

if randomise_positions:
    n_ra, n_dec = np.random.uniform(0, 2*np.pi, len(data)), np.arccos(np.random.uniform(-1,1,len(data)))-np.pi/2
else:
    n_ra, n_dec = data[:,1]*np.pi/180, data[:,2]*np.pi/180      # Convert to rad

n_theta = np.pi/2 - n_dec

if randomise_spins:
    dtheta_true = jnp.pi/2 - ddec_true          # Convert to theta for formula below
    prob = M_true + D_true*(jnp.sin(dtheta_true)*jnp.sin(n_theta)*jnp.cos(dra_true-n_ra) + jnp.cos(dtheta_true)*jnp.cos(n_theta))  # Vector over all gals
    if np.sum(prob<0)>0 or np.sum(prob>1)>0:
        print("Prob is outside of [0,1]")
        quit()
    rand = np.random.uniform(0,1,len(data))
    spin = np.zeros(len(data))
    spin[prob>rand] = 1
else:
    spin = data[:,3]
    spin[spin==-1]=0

if rank==0: print("Number of data points:", len(spin))

rng_key = random.PRNGKey(np.random.randint(1000000))
rng_key, rng_key_ = random.split(rng_key)
kernel = numpyro.infer.NUTS(model, init_strategy=numpyro.infer.initialization.init_to_median(num_samples=20000))

if parallel:
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, num_chains=multiprocessing.cpu_count(), progress_bar=progress)
else:
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=progress)
    
try:
    mcmc.run(rng_key_)
    samples = mcmc.get_samples()
except Exception as ex:
    print("An exception of type {0} occurred. Arguments: {1!r}".format(type(ex).__name__, ex.args), flush=True)
    print("Failing on exception", failed_count, flush=True)
    quit()

keys = list(samples.keys())

labels = []
nparam = np.zeros(len(keys), dtype=int)
    
for m in range(len(keys)):
    if len(samples[keys[m]].shape) == 1:
        labels += [keys[m]]
        nparam[m] = 1
    else:
        nparam[m] = samples[keys[m]].shape[1]
        labels += [keys[m] + '_%i'%n for n in range(nparam[m])]
        
nparam = [0] + list(np.cumsum(nparam))

all_samples = np.empty((samples[keys[0]].shape[0], len(labels)))
for m in range(len(keys)):
    if len(samples[keys[m]].shape) == 1:
        all_samples[:,nparam[m]] = samples[keys[m]][:]
    else:
        for n in range(nparam[m+1]-nparam[m]):
            all_samples[:,nparam[m]+n] = samples[keys[m]][:,n]
            
labels = np.array(labels)
all_samples = np.array(all_samples)

labels_keep = np.array(["M", "D", "d_ra", "d_dec"])

samples_1 = np.copy(all_samples)
for m in range(len(labels)):
    ind2 = np.where(labels_keep==labels[m])[0][0]
    samples_1[:,ind2] = all_samples[:,m]
    
samps_M, samps_D, samps_dra, samps_ddec = samples_1[:,0], samples_1[:,1], samples_1[:,2], samples_1[:,3]

mean_M, mean_D, mean_dra, mean_ddec = np.mean(samps_M), np.mean(samps_D), np.mean(samps_dra), np.mean(samps_ddec)
std_M, std_D, std_dra, std_ddec = np.std(samps_M), np.std(samps_D), np.std(samps_dra), np.std(samps_ddec)

dev_M = (mean_M - M_true)/std_M
dev_D = (mean_D - D_true)/std_D
dev_dra = (mean_dra - dra_true)/std_dra
dev_ddec = (mean_ddec - ddec_true)/std_ddec

#mcmc.print_summary()

labels_show = [r'${\rm M}$', r'${\rm D}$', r'${\rm d_{RA}/rad}$', r'${\rm d_{Dec}/rad}$']

if rank<20:
    plt.clf()
    corner.corner(samples_1, labels=labels_show, show_titles=True, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 12}, title_fmt = '.3f', label_kwargs={"fontsize": 13}, verbose=True)
    if randomise_spins and randomise_positions:
        plt.savefig("Plots/numpyro_corner_"+str(rank)+".png", bbox_inches="tight")
    elif randomise_spins:
        plt.savefig("Plots/numpyro_corner_"+str(rank)+".png", bbox_inches="tight")
    else:
        plt.savefig("Plots/numpyro_corner_"+str(rank)+".png", bbox_inches="tight")

print(rank, "  ", dev_M, dev_D, dev_dra, dev_ddec)

buff = np.array([dev_M, dev_D, dev_dra, dev_ddec])

if rank==0:
    final_arr = np.zeros([size,4])
    final_arr[0] = buff
    for i in range(1, size):
        comm.Recv(buff, source=i)        # Make sure everything's finished before doing the final step
        final_arr[i]=buff
else:
    comm.Send(buff, dest=0)

if rank==0:
    if randomise_spins and randomise_positions:
        if allow_neg_D:
            np.savetxt("Devs_Rspinpos_negD.dat", final_arr)
        else:
            np.savetxt("Devs_Rspinpos.dat", final_arr)
    elif randomise_spins:
        if allow_neg_D:
            np.savetxt("Devs_Rspin_negD.dat", final_arr)
        else:
            np.savetxt("Devs_Rspin.dat", final_arr)
    else:
        if allow_neg_D:
            np.savetxt("Devs_Rpos_negD.dat", final_arr)
        else:
            np.savetxt("Devs_Rpos.dat", final_arr)
    
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)

    dev_M, dev_D, dev_dra, dev_ddec = final_arr[:,0], final_arr[:,1], final_arr[:,2], final_arr[:,3]

    xarr = np.linspace(-5,5,500)
    yarr = 1/np.sqrt(2*np.pi)*np.exp(-xarr**2/2)

    plt.clf()
    sns.kdeplot(dev_M, linewidth=1.5, color="blue", cut=3, bw_adjust=1, label=r'$M$')
    sns.kdeplot(dev_D, linewidth=1.5, color="red", cut=3, bw_adjust=1, label=r'$D$')
    sns.kdeplot(dev_dra, linewidth=1.5, color="green", cut=3, bw_adjust=1, label=r'$d_\alpha$')
    sns.kdeplot(dev_ddec, linewidth=1.5, color="cyan", cut=3, bw_adjust=1, label=r'$d_\delta$')
    plt.plot(xarr, yarr, color="black", linestyle="dashed", label="Std. norm.", alpha=0.5)
    plt.legend(prop={'size': 12})
    plt.xlim([-4,4])
    plt.xlabel("Bias", fontsize=13)

    if randomise_spins and randomise_positions:
        if allow_neg_D:
            plt.savefig("Plots/Devs_Rspinpos_negD.png", bbox_inches="tight")
        else:
            plt.savefig("Plots/Devs_Rspinpos.png", bbox_inches="tight")
    elif randomise_spins:
        if allow_neg_D:
            plt.savefig("Plots/Devs_Rspin_negD.png", bbox_inches="tight")
        else:
            plt.savefig("Plots/Devs_Rspin.png", bbox_inches="tight")
    else:
        if allow_neg_D:
            plt.savefig("Plots/Devs_Rpos_negD.png", bbox_inches="tight")
        else:
            plt.savefig("Plots/Devs_Rpos.png", bbox_inches="tight")
    
