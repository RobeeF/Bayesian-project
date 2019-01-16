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