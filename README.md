# Galaxy Spin Anisotropy

Code for reproducing the results of Patel & Desmond 2024 (https://arxiv.org/abs/2404.06617), assessing the significance of anisotropy in galaxy spin directions. This includes both a dipole and a "hemispherical asymmetry" in which different hemispheres have different spin probabilities, without the cos(theta) dependence of the dipole. Both frequentist and Bayesian codes are provided. Also included is a code mock_data_validation.py which checks that the inference is unbiased on mock data with known injected monopole and dipole. See the comments at the start of the codes for further instructions.

Contact Dhruva Patel (zx970439@ou.ac.uk) and Harry Desmond (harry.desmond@port.ac.uk) with feedback or questions.

------

Also included is a program numpyro_spins_HSC.py for reproducing Stiskalek & Desmond 2024 (https://arxiv.org/abs/2410.18884), assessing the significance of a dipole anisotropy in Hyper Suprime-Cam Data Release 3 (data_hsc.csv, also included). This code uses NumPyro to sample with the No U-Turns Sampler, and the harmonic estimator to calculate evidences.

Contact Richard Stiskalek (richard.stiskalek@physics.ox.ac.uk) and Harry Desmond (harry.desmond@port.ac.uk) with feedback or questions.
