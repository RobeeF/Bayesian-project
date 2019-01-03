# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:09:55 2018

@author: robin
"""

import os

os.chdir("C:/Users/quent/Desktop/3A_ENSAE/Stats Bayesiennes/Projet/bayesian-project-master")


import pandas as pd
import numpy as np
from functions import *

seed = 1712
rnd = np.random.RandomState(seed)

## Import data
X = pd.read_csv('nodal.csv')
del(X['Unnamed: 0'])
del(X['id'])
y = X.iloc[:,0].values
X = X.iloc[:,1:].values

## Specifying the length of the chain and the prior parameters to send to the Gibbs Sampler
d = X.shape[1]

a = np.full(d, 0.75)
A = np.linalg.inv(5*np.identity(d))
B = np.linalg.inv(np.identity(d)) 



iters = 5000
init = {"a": a,
        "A": A}

## specify hyper parameters
hypers = {"SAMPLE_SPACING": 1,
         "BURN_IN": 500,
         "seed": seed}

## Run the Gibbs Sampler (for all the covariates) and compute the marginal likelihood
beta, beta_z, B = GibbsSampler(X, y, iters, init, hypers)
log_marg = compute_marg_likelihood(X, y, iters, init, hypers)