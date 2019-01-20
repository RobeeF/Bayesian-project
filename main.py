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

seed = 1712
rnd = np.random.RandomState(seed)

#======================================================================================
# Nodal Data 
#======================================================================================


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
        'q_params': np.full(d,1), 'A':100}

## specify hyper parameters
hypers = {"SAMPLE_SPACING": 1,
         "BURN_IN": 500,
         "seed": seed}

G = 100

mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, G, init, hypers)
pd.Series(mu[:,0]).plot()
pd.Series(mu[:,2]).plot()

mu_star = np.array(mu).mean(axis=0)
sigma_square_star = np.array(sigma_square).mean(axis=0)
q_star = np.array(q).mean(axis=0)


compute_marg_likelihood_and_NSE_galaxies(y, G, init, hypers)