# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:09:55 2018

@author: robin
"""

import os

os.chdir("C:/Users/robin/Documents/GitHub/bayesian-project")

import pandas as pd
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from copy import deepcopy

seed = 1712
rnd = np.random.RandomState(seed)

#======================================================================================
# Nodal Data 
#======================================================================================

"""
## Import data
X = pd.read_csv('nodal.csv')
del(X['Unnamed: 0'])
del(X['id'])
y = X.iloc[:,0].values
X = X.iloc[:,1:].values
n = X.shape[0]

## specify hyper parameters
hypers = {"SAMPLE_SPACING": 1,
         "BURN_IN": 500,
         "seed": seed}
iters = 5000


## Paper's Models (corresponding columns to keep)
MODELS = {"M1" : [],
          "M2" : [0],
          "M3" : [1],
          "M4" : [2],
          "M5" : [3],
          "M6" : [4],
          "M7" : [1,3],
          "M8" : [1,2,3],
          "M9" : [1,2,3,4]
        }

## log scale of x2
X[:,1]=np.log(X[:,1])


## Run the Gibbs Sampler (for all the models) 
## and compute the marginal log_likelihood and the numerical standard error
RESULTS = dict()

for i in MODELS :

    #Select the variables
    C = np.full((n,1),1)
    X_model = np.concatenate((C,X[:,MODELS[i]]), axis = 1)
    
    ## Specifying the length of the chain and the prior parameters to send to the Gibbs Sampler
    d = X_model.shape[1]
    a = np.full(d, 0.75)
    A = np.linalg.inv(5*np.identity(d))

    init = {"a": a,
            "A": A}

    log_marg,NSE = compute_marg_likelihood_and_NSE(X_model, y, iters, init, hypers)
    RESULTS[i] = [log_marg,NSE]
 """
#======================================================================================
# Galaxies data 
#======================================================================================
from functions import *
from scipy.stats import multivariate_normal, norm, invgamma, dirichlet

y = pd.read_csv('galaxies.csv')
y = np.array(y).reshape(-1,1)
y = y/1000 # Velocity/1000 as in the paper 

d = 3
init = {'d':d,'mu_params': np.array([20,100]), 'sigma_square_params': np.array([6,40]),
        'q_params': np.full(d,1)}

## specify hyper parameters
hypers = {"SAMPLE_SPACING": 1,
         "BURN_IN": 500,
         "seed": None}

G = 100

### Check consistency of the results
mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, G, init, hypers)

pd.Series(mu[:,0]).plot()
pd.Series(sigma_square[:,1]).plot()

mu_star = np.array(mu).mean(axis=0)
sigma_square_star = np.array(sigma_square).mean(axis=0)
q_star = np.array(q).mean(axis=0)

# The following output is erroneous. Function to finish
log_marg_likelihood, NSE = compute_marg_likelihood_and_NSE_galaxies(y, G, init, hypers)

### Evaluate the likelihood and the numerical standard error
RESULTS = []
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
from functions import *

n = 500
mean = [5,12]
sigma_square_param = [0.5, 0.5]
cov = np.diag(sigma_square_param)
y = simul_gaussian_mixture(mean, cov, n)
y = np.array(y).reshape(-1,1)

d = 2
init = {'d':d,'mu_params': np.array([10,50]), 'sigma_square_params': np.array([6,60]),
        'q_params': np.full(d,1)}

## specify hyper parameters
hypers = {"SAMPLE_SPACING": 3,
         "BURN_IN": 500, 
         'seed':None}

G = 50

mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, G, init, hypers)

mu_star = np.array(mu).mean(axis=0)
sigma_square_star = np.array(sigma_square).mean(axis=0)
q_star = np.array(q).mean(axis=0)

pd.Series(mu[:,0]).plot()
pd.Series(mu[:,1]).plot()


pd.Series(sigma_square[:,0]).plot()
pd.Series(sigma_square[:,1]).plot()

# Test distance to true parameters
mu_dist, sigma_dist = distance_to_true_params_value(G, mu, sigma_square, mean, sigma_square_param)
pd.Series(mu_dist).plot()
pd.Series(sigma_dist).plot()