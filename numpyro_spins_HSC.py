# Copyright (C) 2024 Harry Desmond, Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Code to infer the monopole, dipole and quadrupole of binary galaxy spins on the sky.
Set up to analyse HSC DR3 (see Stiskalek & Desmond 2024).
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from h5py import File
import harmonic as hm
from jax import random
from jax.lax import cond
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import ProjectedNormal, Uniform
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, init_to_sample
from numpyro.infer.reparam import ProjectedNormalReparam
import scienceplots

###############################################################################
#                           Plotting routines                                 #
###############################################################################


def plot_sample_distribution(RA, dec, data_fname, fdir, ext="png",
                             verbose=True):
    fname = join(fdir, f"{Path(data_fname).stem}_sky_dist.{ext}")

    with plt.style.context("science"):
        hp.mollview(title=None, coord=['C'], notext=True)

        theta = np.radians(90 - dec)
        phi = np.radians(RA)
        hp.projscatter(theta, phi, lonlat=False, s=1.)

        # Add grid and legend
        hp.graticule()

        plt.tight_layout()
        if verbose:
            print(f"Saving a sky distribution plot to `{fname}`.")
        plt.savefig(fname, dpi=450)
        plt.close()


def labels2pretty(labels):
    x = {"M": r"$M$",
         "D": r"$D$",
         "dec_dipole": r"$\mathrm{Dec}$",
         "ra_dipole": r"$\mathrm{RA} + \pi$",
         }

    return [x.get(label, label) for label in labels]


def plot_corner(samples, data_fname, fdir, exclude_dipole, include_quadrupole,
                ext="png", verbose=True):
    fname = join(fdir, f"{Path(data_fname).stem}_corner.{ext}")

    if exclude_dipole:
        fname = fname.replace(f".{ext}", f"_nodipole.{ext}")

    if include_quadrupole:
        fname = fname.replace(f".{ext}", f"_quadrupole.{ext}")

    labels_keep = ["M",
                   "D", "dec_dipole", "ra_dipole",
                   "Q", "q1_ra", "q1_dec", "q2_ra", "q2_dec"]

    data, keys = [], []
    for i, label in enumerate(labels_keep):
        if label in samples:
            data.append(samples[label])
            keys.append(label)

    labels = labels2pretty(keys)
    data = np.asarray(data).T

    if not exclude_dipole:
        k = keys.index("ra_dipole")
        data[:, k] += np.pi
        data[:, k] %= 2 * np.pi

    with plt.style.context("science"):
        fig = corner(data, labels=labels, show_titles=True, smooth=1,
                     smooth1d=True, title_fmt='.3f')

        for ax in fig.get_axes():
            ax.tick_params(labelsize=17)
            ax.xaxis.label.set_size(17)
            ax.yaxis.label.set_size(17)

            if ax.title:
                ax.title.set_fontsize(17)

        if verbose:
            print(f"Saving a corner plot to `{fname}`.")

        fig.savefig(fname, dpi=450)
        plt.close()


###############################################################################
#                              Load data                                      #
###############################################################################


def read_data(fname, plots_dir, make_plot=True, ext="png"):
    data = np.genfromtxt(fname, delimiter=",", skip_header=1)
    data = data[:-3]
    n_ra, n_dec, z, spin = [data[:, i] for i in (0, 1, 2, 3)]
    spin = (spin == 1).astype(float)
    print(f"Loaded {len(data)} data points.")

    m = np.isfinite(n_ra) & np.isfinite(n_dec) & np.isfinite(spin)
    print(f"Masking {np.sum(~m)} non-finite data point(s).")
    m = np.isfinite(n_ra) & np.isfinite(n_dec) & np.isfinite(spin)
    n_ra, n_dec, z, spin = n_ra[m], n_dec[m], z[m], spin[m]

    if make_plot:
        plot_sample_distribution(n_ra, n_dec, fname, plots_dir, ext=ext)

    # Convert RA/dec to radians and theta/phi
    n_phi = np.deg2rad(n_ra)
    n_theta = np.pi / 2 - np.deg2rad(n_dec)

    return n_phi, n_theta, z, spin


def save_samples(samples, ln_evidence, err_ln_evidence, data_fname, res_dir,
                 exclude_dipole, include_quadrupole, log_posterior=None):
    fname = join(res_dir, f"{Path(data_fname).stem}_corner.hdf5")

    if exclude_dipole:
        fname = fname.replace(".hdf5", "_nodipole.hdf5")

    if include_quadrupole:
        fname = fname.replace(".hdf5", "_quadrupole.hdf5")

    print(f"Saving samples to `{fname}`.")
    with File(fname, 'w') as f:
        grp = f.create_group("samples")
        for key in samples.keys():
            grp.create_dataset(key, data=samples[key])

        f.attrs["ln_evidence"] = ln_evidence
        f.attrs["err_ln_evidence"] = err_ln_evidence

        if log_posterior is not None:
            f.create_dataset("log_posterior", data=log_posterior)

###############################################################################
#                       Harmonic evidence routines                            #
###############################################################################


def get_harmonic_evidence(samples, log_posterior):
    """Compute evidence using the `harmonic` package."""
    nchains_harmonic = 10
    epochs_num = 50

    # Turn the samples to an array that harmonic expects.
    keys = ["M", "D", "Q", "dec_dipole", "ra_dipole",
            "q1_dec", "q1_ra", "q2_dec", "q2_ra"]
    data, keys_accepted = [], []
    for key in keys:
        if key in samples:
            data.append(samples[key])
            keys_accepted.append(key)
    data = np.asarray(data).T

    print(f"Calculating the evidence with keys: {keys_accepted}")

    # Reshape the data and log posterior to split over chains.
    data = data.reshape(nchains_harmonic, -1, data.shape[-1])
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    # Do some standard checks of inputs.
    if data.ndim != 3:
        raise ValueError("The samples must be a 3-dimensional array of shape `(nchains, nsamples, ndim)`.")

    if log_posterior.ndim != 2 and log_posterior.shape[:2] != data.shape[:2]:
        raise ValueError("The log posterior must be a 2-dimensional array of shape `(nchains, nsamples)`.")

    ndim = data.shape[-1]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(data, log_posterior)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=0.5)

    # Use default hyperparameters
    model = hm.model.RQSplineModel(ndim, standardize=True, temperature=0.8)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=True)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    ln_evidence = -ev.ln_evidence_inv
    err_ln_evidence = ev.compute_ln_inv_evidence_errors()

    return ln_evidence, err_ln_evidence

###############################################################################
#                        NumPyro probabilistic model                          #
###############################################################################


def sample_sky_direction(name):
    with reparam(config={f"xdir_dipole_{name}": ProjectedNormalReparam()}):
        xdir = sample(f"xdir_dipole_{name}", ProjectedNormal(jnp.zeros(3)))

    dec = deterministic(f"dec_{name}", jnp.pi / 2 - jnp.arccos(xdir[2]))
    ra = jnp.arctan2(xdir[1], xdir[0])
    ra = cond(ra < 0, lambda x: x + 2 * jnp.pi, lambda x: x, ra)
    ra = deterministic(f"ra_{name}", ra)

    return ra, dec


def sample_quadrupole_direction():
    ra1 = sample("q1_ra", Uniform(0, 2 * jnp.pi))
    ra2 = sample("q2_ra", Uniform(ra1, 2 * jnp.pi))

    sin_dec1 = sample("sin_dec_q1", Uniform(-1, 1))
    sin_dec2 = sample("sin_dec_q2", Uniform(-1, 1))

    dec1 = deterministic("q1_dec", jnp.arcsin(sin_dec1))
    dec2 = deterministic("q2_dec", jnp.arcsin(sin_dec2))

    return ra1, dec1, ra2, dec2


def log_prob(M, D, d_ra, d_dec, Q, q1_ra, q1_dec, q2_ra, q2_dec, n_phi,
             n_theta, spin):
    d_theta = 0.5 * jnp.pi - d_dec
    q1_theta = 0.5 * jnp.pi - q1_dec
    q2_theta = 0.5 * jnp.pi - q2_dec

    prob = (+ M + D * (jnp.sin(d_theta) * jnp.sin(n_theta) * jnp.cos(d_ra - n_phi) + jnp.cos(d_theta) * jnp.cos(n_theta)) + Q * ((jnp.sin(q1_theta) * jnp.sin(n_theta) * jnp.cos(q1_ra - n_phi) + jnp.cos(q1_theta) * jnp.cos(n_theta)) * (jnp.sin(q2_theta) * jnp.sin(n_theta) * jnp.cos(q2_ra - n_phi) + jnp.cos(q2_theta) * jnp.cos(n_theta)) - 1/3 * (jnp.sin(q1_theta) * jnp.sin(q2_theta) * jnp.cos(q1_ra - q2_ra) + jnp.cos(q1_theta) * jnp.cos(q2_theta))))
    lnlike = jnp.log(prob)*spin + jnp.log(1 - prob) * (1 - spin)

    return lnlike


def model(n_phi, n_theta, spin, exclude_dipole, sample_quadrupole):
    M = sample("M", Uniform(0., 1.0))

    if exclude_dipole:
        D, d_ra, d_dec = 0., 0., 0.
    else:
        D = sample("D", Uniform(0, 0.1))
        d_ra, d_dec = sample_sky_direction("dipole")

    if sample_quadrupole:
        Q = sample("Q", Uniform(0, 0.25))
        q1_ra, q1_dec, q2_ra, q2_dec = sample_quadrupole_direction()
    else:
        Q = 0.
        q1_ra, q1_dec, q2_ra, q2_dec = 0., 0., 0., 0.

    with plate("data", len(n_phi)):
        ll = log_prob(M, D, d_ra, d_dec, Q, q1_ra, q1_dec, q2_ra, q2_dec,
                      n_phi, n_theta, spin)
        factor("obs", ll,)


###############################################################################
#                         Command line interface                              #
###############################################################################


def run_single(nwarm, nsamp, n_phi, n_theta, spin, exclude_dipole,
               include_quadrupole):
    kernel = NUTS(model, init_strategy=init_to_sample())
    mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp)

    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, extra_fields=("potential_energy",), n_phi=n_phi,
             n_theta=n_theta, spin=spin,
             exclude_dipole=exclude_dipole,
             sample_quadrupole=include_quadrupole)
    mcmc.print_summary(exclude_deterministic=False)

    return mcmc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exclude_dipole", action="store_true",
                        default=False)
    parser.add_argument("--include_quadrupole", action="store_true",
                        default=False)

    args = parser.parse_args()

    if args.exclude_dipole:
        print("Excluding the dipole from the model.")

    if args.include_quadrupole:
        print("Including the quadrupole in the model.")

    nwarm = 1000
    nsamp = 9000

    fname = "data_hsc.csv"
    plots_dir = "./"
    res_dir = "./"
    n_phi, n_theta, z, spin = read_data(fname, plots_dir)

    print(f"The average spin is {np.mean(spin)}")

    mcmc = run_single(nwarm, nsamp, n_phi, n_theta, spin,
                        args.exclude_dipole, args.include_quadrupole)
    samples = mcmc.get_samples()
    log_posterior = -mcmc.get_extra_fields()["potential_energy"]

    plot_corner(samples, fname, plots_dir, args.exclude_dipole,
                args.include_quadrupole, ext="pdf")

    if not args.exclude_dipole:
        ln_evidence, err_ln_evidence = get_harmonic_evidence(
            samples, log_posterior)
    else:
        ln_evidence, err_ln_evidence = np.nan, np.nan

    print(f"Ln(Z)     = {ln_evidence}")
    print(f"err ln(Z) = {err_ln_evidence}")

    save_samples(samples, ln_evidence, err_ln_evidence, fname, res_dir,
                    args.exclude_dipole, args.include_quadrupole,
                    log_posterior=log_posterior)
