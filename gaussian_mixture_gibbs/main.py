# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:09:55 2018

@author: robin
"""

import os

os.chdir("C:/Users/robin/Documents/GitHub/bayesian-project/gaussian_mixture_gibbs")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from functions import *

#======================================================================================
# Galaxies data 
#======================================================================================

y = pd.read_csv('galaxies.csv')
y = np.array(y).reshape(-1,1)
y = y/1000 # Velocity/1000 as in the paper 

d = 3
init = {'d':d,'mu_params': np.array([20,100]), 'sigma_square_params': np.array([6,40]),
        'q_params': np.full(d,1)}

# Specify hyper parameters
hypers = {"SAMPLE_SPACING": 1,
         "BURN_IN": 500,
         "seed": None}

G = 100

# Run the sampler
mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, G, init, hypers)

pd.Series(mu[:,0]).plot()
pd.Series(sigma_square[:,1]).plot()

mu_star = np.array(mu).mean(axis=0)
sigma_square_star = np.array(sigma_square).mean(axis=0)
q_star = np.array(q).mean(axis=0)

# The following output is erroneous. Function to finish
log_marg_likelihood, NSE = compute_marg_likelihood_and_NSE_galaxies(y, G, init, hypers)

# Evaluate the likelihood and the numerical standard error
RESULTS = dict()
for d in range(1,6): # Evaluate the model for 1,2,3,4 or 5 components
    init['d'] = d
    init['q_params'] = np.full(d,1)
    log_marg,NSE = compute_marg_likelihood_and_NSE_galaxies(y, G, init, hypers)
    RESULTS[d] = [log_marg,NSE]


# Graph for the chosen model (out of the 5 runned above)
sample = deepcopy(y)
sample.sort()
x = np.linspace(-3,60,10000)

for i in range(d):
    gaussian_pdf = norm.pdf(x, loc=mu_star[i], scale=sigma_square_star[i])    # for example
    plt.plot(x, gaussian_pdf)
plt.plot(sample,np.full(82,0), 'r^') # Could compute which point belongs to which gaussian and print them in the gaussian colors

plt.show()

#======================================================================================
# Synthetic data 
#======================================================================================

n = 500 # A bigger sample size is chosen to have more precise results
mean = [5,12]
sigma_square_param = [0.5, 0.5]
cov = np.diag(sigma_square_param)
y = simul_gaussian_mixture(mean, cov, n)
y = np.array(y).reshape(-1,1)

d = 2
init = {'d':d,'mu_params': np.array([10,50]), 'sigma_square_params': np.array([6,60]),
        'q_params': np.full(d,1)}

# specify hyper parameters
hypers = {"SAMPLE_SPACING": 3,
         "BURN_IN": 500, 
         'seed':None}

G = 500

# Run the sampler
mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, G, init, hypers)

# Compute Theta*
mu_star = np.array(mu).mean(axis=0)
sigma_square_star = np.array(sigma_square).mean(axis=0)
q_star = np.array(q).mean(axis=0)

# Plot the chain for the first and the second gaussian
pd.Series(mu[:,0]).plot()
pd.Series(mu[:,1]).plot()

pd.Series(sigma_square[:,0]).plot()
pd.Series(sigma_square[:,1]).plot()

# Compute the sequence of distances between estimated and true parameters at each iteration
mu_dist, sigma_dist = distance_to_true_params_value(G, mu, sigma_square, mean, sigma_square_param)

# Plot the distance sequence
pd.Series(mu_dist).plot()
pd.Series(sigma_dist).plot()