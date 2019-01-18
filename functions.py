# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:03:26 2018

@author: robin
"""

import numpy as np
import numpy.linalg
import os 
import copy
from scipy.stats import multivariate_normal, invgamma, dirichlet, multinomial, norm
import pandas as pd
import statsmodels.discrete.discrete_model as sm


def compute_beta_prior(mean, cov, seed=None):
    ''' Generate draws from the multivariate normal prior 
    mean: (array-like) Mean of the multivariate prior
    cov: (ndarray) Covariance of the multivariate prior
    
    returns: (array-like) the draws from the prior
    '''
    rnd = np.random.RandomState(seed)
    return rnd.multivariate_normal(mean=mean, cov=cov)

def compute_z(beta, X, y,seed=None):
    ''' Simulate z_i the hidden variables of the model from a multivariate gaussian distribution
    beta: (array-like) the coefficients estimated from the probit model
    X: (ndarray) exogeneous variables of the model
    y: (ndarray) endogeneous variable of the model
    seed: (int) random seed of the model to have reproducible results

    returns: (array-like) the draws of z_i simulated in the Gibbs Sampler
    '''
    
    rnd = np.random.RandomState(seed)
    xb = np.dot(X,beta.reshape(-1,1))
    n = len(y)
    z = np.zeros(shape=(n,1))
    
    # Simulate a gaussian for each i
    gaussian = rnd.multivariate_normal(mean=xb[:,0],cov=np.identity(n))
    # Troncate the observations according to y value
    z=np.where(y==np.zeros(n),np.minimum(np.zeros(n),gaussian), np.maximum(np.zeros(n),gaussian)) #MODIF : changement ordre min/max
    z = z.reshape(-1,1)
    return z

def compute_beta_z(z,X,A,a):
    ''' Simulate beta_z from the draws of hidden variables z_i 
    z: (array-like) hidden variables of the model
    X: (ndarray) exogeneous variables of the model
    y: (ndarray) endogeneous variable of the model
    A: (ndarray) Variance-covariance matrix of the multivariate prior pdf
    a: (array-like) mean of the multivariate prior pdf
    
    returns: (array-like) the beta_z simulated in the Gibbs Sampler
    '''
    return np.dot(numpy.linalg.inv(A+ np.dot(X.T,X)),
               np.dot(A,a)+np.dot(X.T,z)[:,0])
    
def compute_B(A,X): # Cheucheu peut-être
    ''' Compute the variance-covariance matrix of the posterior 
    A: (ndarray) Covariance of the prior
    X: (ndarray) Exogeneous variables of the model
    
    returns: (ndarray) Cov-Var matrix of the posterior 
    '''
    return np.linalg.inv(A + np.dot(X.T,X))

def compute_Omega(s,h):
    ''' Compute Omega_s (from part 3 of the article)
    Warning : This computation works only in the 2.1.1 case
    
    s: (integer) parameter of Omega
    h: (array-like) Conditional densities as defined in 3. pi(Beta*|y,z^g)_g
    
    returns : (float because we are in case 2.1.1)  Coefficient Omega used to compute NSE
    '''
    h_hat = h.mean()
    G = np.shape(h)[0]
    h = h[s:]
    h = h-h_hat
    return (1/G)*np.sum(h*h)

def compute_var_h(h,q=10):
    ''' Compute var(h) as in last formule of p.4 right column (for case 2.1.1)
    h: (array-like) Conditional densities as defined in 3. pi(Beta*|y,z^g)_g
    q: (integer) parameter q=10 by default (as suggered in the article)
    
    returns (float because we are in case 2.1.1) var(h_hat)
    '''
    G = np.shape(h)[0]
    temp = np.array([(1-s/(q+1))*2*compute_Omega(s,h) for s in range(1,q+1)])
    return (1/G)*(compute_Omega(0,h)+np.sum(temp))
    
    
    
  
def GibbsSampler(X, y, iters, init, hypers, seed=None): 
    ''' Gibbs sampler applied to the nodal set from  Chib (1995).
    X: (ndarray) exogeneous variables
    y : (array-like) endogeneous variables
    iters: (int) length of the MCMC
    init: (array-like) initialisation parameters
    hypers: (array-like) hyper-parameters
    
    returns: (tuple) the simulated beta chain (array-like), the b_z chain (array-like) as a by product and B the covariance matrix of the posterior (ndarray)
    '''
    
    # Initialisation
    a,A = init['a'], init['A']
    
    # Hyper-parameters
    BURN_IN = hypers['BURN_IN']
    SAMPLE_SPACING = hypers['SAMPLE_SPACING']
    seed = hypers['seed']

    rnd = np.random.RandomState(seed)
    beta = compute_beta_prior(mean=a,cov=np.linalg.inv(A),seed=seed) # MODIF : np.linalg.inv(A) in place of A
    z = compute_z(beta,X,y,seed)
    B = compute_B(A,X)

    
    # We wait untill BURN_IN updates before sampling 
    # Then we sample every SAMPLE_SPACING iterations
    # As a result p*SAMPLE_SPACING + BURN_IN iterations are needed 
    remaining_iter = iters*SAMPLE_SPACING + BURN_IN 

    sample_beta = [] # Will contain the sampled betas
    sample_beta_z = [] # Will contain the sampled beta_Zs 

    while remaining_iter>0: 
        beta_z = compute_beta_z(z,X,A,a) # beta_z are updated
        beta = rnd.multivariate_normal(mean=beta_z, cov=B) # beta updated
        z = compute_z(beta,X,y,seed) # MODIF : adds z updated
        
        if remaining_iter%SAMPLE_SPACING == 0 and BURN_IN <=0: # If the BURN_IN period is over
            # and that we need to sample this iteration
            sample_beta.append(copy.deepcopy(beta))
            sample_beta_z.append(copy.deepcopy(beta_z))
                               
        BURN_IN-=1
        remaining_iter-=1
        
    if iters == 1: # If there is only one observation
        # return the observation, not a list of one element
        return sample_beta[0], sample_beta_z[0], B
    else:
        return sample_beta,sample_beta_z, B 
    
def compute_marg_likelihood_and_NSE(X, y, iters, init, hypers):
    ''' Compute the marginal likelihood from the Gibbs Sampler output according to Chib (1995)
    X: (ndarray) exogeneous variables
    y : (array-like) endogeneous variables
    iters: (int) length of the MCMC
    init: (array-like) initialisation parameters
    hypers: (array-like) hyper-parameters
    
    returns: (float) the marginal likelihood/normalizing constant 
    '''
    
    # Initialisation
    a,A = init['a'], init['A']
        
    beta, beta_z, B = GibbsSampler(X, y, iters, init, hypers)

    beta_star = np.array(beta).mean(axis=0)
    beta_z = np.array(beta_z)
    
    ## Marginal likelihood computation P7, right column
    # First term:
    log_like= sm.Probit(endog=y, exog=X).loglike(params=beta_star)
    # Second term
    prior = multivariate_normal.logpdf(x=beta_star, mean=a, cov=A)
    multivariate_normal.pdf(x=beta_star, mean=a, cov=A)
    # Third term
    conditional_densities = np.array([multivariate_normal.pdf(x=beta_star, mean=beta_z[i], cov=B) for i in range(iters)])
    posterior = np.log(conditional_densities.mean()) 
    # pdf renvoie un gros nombre...: Compréhension de liste peut être amélioré
    # Marginal likelihood
    log_marg_likelihood = log_like + prior - posterior
    
    #Numerical Standard Error
    NSE = np.sqrt(compute_var_h(conditional_densities,q=10)/(conditional_densities.mean()**2))
    
    return log_marg_likelihood, NSE


#======================================================================================
# Galaxies functions
#======================================================================================


def compute_prior_galaxies(mu_params, sigma_params,q_params, d, seed=None):
    ''' Generate draws from the multivariate normal prior 
    mean: (array-like) Mean of the multivariate prior
    cov: (ndarray) Covariance of the multivariate prior
    
    returns: (array-like) the draws from the prior
    '''
    rnd = np.random.RandomState(seed)
    mu =rnd.normal(loc=mu_params[0], scale=np.sqrt(mu_params[1]), size=d)
    sigma_square = invgamma.rvs(a=sigma_params[0]/2,scale=sigma_params[1]/2,size=d)
    q = dirichlet.rvs(alpha=q_params, random_state=seed)[0]
    
    return mu, sigma_square, q

def compute_z_galaxies(q):
    q[q < 0] = 0 # Quick fix: sometimes the number returned by the gaussian is negative and then the code crashes
    p = q/q.sum()
    draws = multinomial.rvs(n=1, p=p, size=82) # 82 hardcoded
    z = np.where(draws==1)[1]
    return z, np.stack(draws)

#q = np.array([1]*3)
#compute_z_galaxies(q)[0]

def compute_B_galaxies(A, sigma_square, n,d):
    B = np.reciprocal(np.full(d,A)+np.power(sigma_square,[-1]*d)*n)
    return B


def compute_n(z,d): # Très sale
    count = numpy.unique(z, return_counts=True)
    if len(count[0])<d: # Sometimes some groups are empty need to fill with zeros.. 
        n = []
        for i in range(0,d):
            if i in count[0]:
                n.append(count[1][numpy.where(count[0]==i)][0])
            else:
                n.append(0)
        return n
    else:
        return count[1]

z = np.array([4,1,1,1,2])
compute_n(z,5)


def GibbsSampler2(y, iters, init, hypers, seed=None): 
    ''' Gibbs sampler applied to the nodal set from  Chib (1995).
    X: (ndarray) exogeneous variables
    y : (array-like) endogeneous variables
    iters: (int) length of the MCMC
    init: (array-like) initialisation parameters
    hypers: (array-like) hyper-parameters
    
    returns: (tuple) the simulated beta chain (array-like), the b_z chain (array-like) as a by product and B the covariance matrix of the posterior (ndarray)
    '''
    
    # Initialisation
    A,d =  init['A'], init['d']

    mu_params, sigma_square_params, q_params = init['mu_params'], init['sigma_square_params'], init['q_params']
    
    # Hyper-parameters
    BURN_IN = hypers['BURN_IN']
    SAMPLE_SPACING = hypers['SAMPLE_SPACING']
    seed = hypers['seed']

    rnd = np.random.RandomState(seed)
    mu, sigma_square, q = compute_prior_galaxies(mu_params, sigma_square_params,q_params,d)
    z, i = compute_z_galaxies(q) # Split in to functions ?
    delta = np.diag(np.dot((y*i - np.multiply(i,mu)).T,(y*i - np.multiply(i,mu))))

    # We wait untill BURN_IN updates before sampling 
    # Then we sample every SAMPLE_SPACING iterations and we sample d times more to evaluate by block (G*d)
    # As a result p*SAMPLE_SPACING + BURN_IN iterations are needed 
    remaining_iter = iters*SAMPLE_SPACING*d + BURN_IN 

    sample_mu = [] # Will contain the sampled betas
    sample_sigma_square = [] # Will contain the sampled beta_Zs 
    sample_q = [] # Will contain the sampled beta_Zs 

    while remaining_iter>0: 
        n = compute_n(z,d)
        B = compute_B_galaxies(A, sigma_square, n,d)
        mu_hat = B*(A*mu_params[0]+np.linalg.multi_dot([np.power(sigma_square,[-1]*d),i.T,y])) # change 1 with i
        
        mu = rnd.multivariate_normal(mean=mu_hat, cov=np.diag(B)) # Why is there a product
        sigma_square = invgamma.rvs(a=(sigma_square_params[0]+n)/2,scale=(sigma_square_params[1]+delta)/2)
        q = dirichlet.rvs(alpha=q_params+n, random_state=seed)[0]

        z,i = compute_z_galaxies(q*rnd.multivariate_normal(mean=mu, cov=np.diag(sigma_square))) # Pas sur pour la multiplication par la normale
        delta = np.diag(np.dot((y*i - np.multiply(i,mu)).T,(y*i - np.multiply(i,mu))))

        if remaining_iter%SAMPLE_SPACING == 0 and BURN_IN <=0: # If the BURN_IN period is over
            # and that we need to sample this iteration
            sample_mu.append(copy.deepcopy(mu))
            sample_sigma_square.append(copy.deepcopy(sigma_square))
            sample_q.append(copy.deepcopy(q))
                               
        BURN_IN-=1
        remaining_iter-=1
        
    if iters == 1: # If there is only one observation
        # return the observation, not a list of one element
        return sample_mu[0], sample_sigma_square[0], sample_q[0] # z aussi non ?
    else:
        return np.stack(sample_mu), np.stack(sample_sigma_square), np.stack(sample_q)