# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:03:26 2018

@author: robin
"""

import numpy as np
import numpy.linalg
import copy
from scipy.stats import multivariate_normal, invgamma, dirichlet, multinomial, norm
import statsmodels.discrete.discrete_model as sm
from scipy.spatial.distance import cdist


def compute_beta_prior(mean, cov, seed=None):
    ''' Generate draws from the multivariate normal prior 
    mean (array-like): Mean of the multivariate prior
    cov (ndarray): Covariance of the multivariate prior
    
    returns (array-like): the draws from the prior
    '''
    rnd = np.random.RandomState(seed)
    return rnd.multivariate_normal(mean=mean, cov=cov)

def compute_z(beta, X, y,seed=None):
    ''' Simulate z_i the hidden variables of the model from a multivariate gaussian distribution
    beta (array-like): the coefficients estimated from the probit model
    X (ndarray): exogeneous variables of the model
    y (ndarray): endogeneous variable of the model
    seed (int): random seed of the model to have reproducible results

    returns (array-like): the draws of z_i simulated in the Gibbs Sampler
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
    z (array-like): hidden variables of the model
    X (ndarray): exogeneous variables of the model
    y (ndarray): endogeneous variable of the model
    A (ndarray): Variance-covariance matrix of the multivariate prior pdf
    a (array-like): mean of the multivariate prior pdf
    
    returns: (array-like) the beta_z simulated in the Gibbs Sampler
    '''
    return np.dot(numpy.linalg.inv(A+ np.dot(X.T,X)),
               np.dot(A,a)+np.dot(X.T,z)[:,0])
    
def compute_B(A,X): # Cheucheu peut-être
    ''' Compute the variance-covariance matrix of the posterior 
    A (ndarray): Covariance of the prior
    X (ndarray): Exogeneous variables of the model
    
    returns (ndarray): Cov-Var matrix of the posterior 
    '''
    return np.linalg.inv(A + np.dot(X.T,X))

def compute_Omega(s,h):
    ''' Compute Omega_s (from part 3 of the article)
    Warning : This computation works only in the 2.1.1 case
    
    s (integer): parameter of Omega
    h (array-like): Conditional densities as defined in 3. pi(Beta*|y,z^g)_g
    
    returns (float because we are in case 2.1.1):  Coefficient Omega used to compute NSE
    '''
    h_hat = h.mean()
    G = np.shape(h)[0]
    h = h[s:]
    h = h-h_hat
    return (1/G)*np.sum(h*h)

def compute_var_h(h,q=10):
    ''' Compute var(h) as in last formule of p.4 right column (for case 2.1.1)
    h (array-like): Conditional densities as defined in 3. pi(Beta*|y,z^g)_g
    q (integer): parameter q=10 by default (as suggered in the article)
    
    returns (float because we are in case 2.1.1) var(h_hat)
    '''
    G = np.shape(h)[0]
    temp = np.array([(1-s/(q+1))*2*compute_Omega(s,h) for s in range(1,q+1)])
    return (1/G)*(compute_Omega(0,h)+np.sum(temp))
    
    
    
  
def GibbsSampler(X, y, iters, init, hypers, seed=None): 
    ''' Gibbs sampler applied to the nodal set from  Chib (1995).
    X (ndarray): exogeneous variables
    y (array-like): endogeneous variables
    iters (int): length of the MCMC
    init (dict): initialisation parameters
    hypers (array-like): hyper-parameters
    
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
    X (ndarray): exogeneous variables
    y (array-like): endogeneous variables
    iters (int): length of the MCMC
    init (dict): initialisation parameters
    hypers (array-like): hyper-parameters
    
    returns (float): the marginal likelihood/normalizing constant 
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

def compute_z_galaxies(q,N):
    q[q < 0] = 0 # Quick fix: sometimes the number returned by the gaussian is negative and then the code crashes
    p = q/q.sum()
    draws = multinomial.rvs(n=1, p=p, size=N) # 82 hardcoded
    z = np.where(draws==1)[1]
    return z, np.stack(draws)

def compute_conditional_z(q,y,mu,sigma_square): #très mauvais, à améliorer
    n = np.shape(y)[0]
    d = np.shape(mu)[0]
    z=np.empty(shape=(n))
    i=np.empty(shape=(n,d))
    for l in range(n):
        temp = np.empty(shape=(d))
        for j in range(d):
            temp[j] = q[j]*multivariate_normal.pdf(y[l,0], mean=mu[j], cov=sigma_square[j])
        temp[temp<0] = 0
        temp=temp/np.sum(temp)
        i[l,:] = multinomial.rvs(n=1, p=temp, size=1)[0]
    z = np.where(i==1)[1]
    return z, i
        
        

def compute_B_galaxies(A, sigma_square, n,d):
    B = np.reciprocal(np.full(d,1/A)+np.reciprocal(sigma_square)*n)
    return B


def compute_n(i):
    return np.apply_along_axis(np.sum,0,i)


def compute_Omega(h,h_hat,s):
    
    n = np.shape(h_hat)[0]
    G = np.shape(h)[1]
    
    temp = (h-h_hat.reshape(-1,1))
    Omega = np.zeros((n,n))

    for i in range(G):
        if i>=s:
            Omega += np.dot(temp[:,i].reshape(-1,1),temp[:,i].reshape(1,-1))

    return Omega/G


def compute_var_h_hat(h,h_hat,q=10):
    
    G = np.shape(h)[1]
    
    var = compute_Omega(h,h_hat,0)
    
    for i in range(1,q+1):
        Omega_s = compute_Omega(h,h_hat,i)
        var += (1-i/(q+1))*(Omega_s+Omega_s.T)
    
    return var/G
        
        
    
    
    
    
    


def GibbsSampler_galaxies(y, iters, init, hypers, seed=None):
    ''' Gibbs sampler applied to the nodal set from  Chib (1995).
    y (array-like): endogeneous variables
    iters (int): length of the MCMC
    init (dict): initialisation parameters
    hypers (array-like): hyper-parameters
    
    returns: (tuple) the simulated beta chain (array-like), the b_z chain (array-like) as a by product and B the covariance matrix of the posterior (ndarray)
    '''
    
    # Initialisation
    d =  init['d']
    N=len(y)

    mu_params, sigma_square_params, q_params = init['mu_params'], init['sigma_square_params'], init['q_params']
    A = mu_params[1]
    
    # Hyper-parameters
    BURN_IN = hypers['BURN_IN']
    SAMPLE_SPACING = hypers['SAMPLE_SPACING']
    seed = hypers['seed']

    rnd = np.random.RandomState(seed)
    mu, sigma_square, q = compute_prior_galaxies(mu_params, sigma_square_params,q_params,d)
    z, i = compute_z_galaxies(q,N) 
    delta = np.apply_along_axis(lambda x : np.sum(x**2), 0, (y*i - np.multiply(i,mu))) 

    sample_mu = []
    sample_mu_hat = []
    sample_B = []
    
    
    Selected_indexes = [SAMPLE_SPACING*l+BURN_IN for l in range(iters)]
    print('mu_hat')
    for l in range(iters*SAMPLE_SPACING + BURN_IN):
        
        n = compute_n(i)
        B = compute_B_galaxies(A, sigma_square, n,d)
        
        mu_hat = B*((1/A)*mu_params[0]+np.reciprocal(sigma_square)*np.apply_along_axis(np.sum,0,i*y))
        mu = rnd.multivariate_normal(mean=mu_hat, cov=np.diag(B)) 
                
        sigma_square = invgamma.rvs(a=(sigma_square_params[0]+n)/2,scale=(sigma_square_params[1]+delta)/2)
        
        q = dirichlet.rvs(alpha=q_params+n, random_state=seed)[0]
                
        z,i = compute_conditional_z(q,y,mu,sigma_square)
        delta = np.apply_along_axis(lambda x : np.sum(x**2), 0, (y*i - np.multiply(i,mu)))

        sample_mu.append(copy.deepcopy(mu))
        sample_mu_hat.append(copy.deepcopy(mu_hat))
        sample_B.append(copy.deepcopy(B))
    

    sample_mu = np.array(sample_mu)[Selected_indexes]
    sample_mu_hat = np.array(sample_mu_hat)[Selected_indexes]
    sample_B = np.array(sample_B)[Selected_indexes]
    
    mu_star = np.array(sample_mu).mean(axis=0) # Take mu=mu*
    # ESTIMER pi_hat(mu_star) mean pi(mu*|y,z_gibb,sigma_square_gibb)

    
    sample_sigma_square = []
    sample_delta = []
    sample_n_for_estim_sigma = []
    print('sigma')
    Selected_indexes = [SAMPLE_SPACING*l for l in range(iters)] # No burn-in for the other params

    for l in range(iters*SAMPLE_SPACING):
        
        n = compute_n(i)
        B = compute_B_galaxies(A,sigma_square, n,d)
        
        sigma_square = invgamma.rvs(a=(sigma_square_params[0]+n)/2,scale=(sigma_square_params[1]+delta)/2)            
        q = dirichlet.rvs(alpha=q_params+n, random_state=seed)[0]
            
        z,i = compute_conditional_z(q,y,mu_star,sigma_square)
        delta = np.apply_along_axis(lambda x : np.sum(x**2), 0, (y*i - np.multiply(i,mu_star)))
            
        sample_sigma_square.append(copy.deepcopy(sigma_square))
        sample_delta.append(copy.deepcopy(delta))
        sample_n_for_estim_sigma.append(copy.deepcopy(n)) 
    
    
    sample_sigma_square = np.array(sample_sigma_square)[Selected_indexes]
    sample_delta = np.array(sample_delta)[Selected_indexes]
    sample_n_for_estim_sigma = np.array(sample_n_for_estim_sigma)[Selected_indexes]
            
    sigma_square_star = np.array(sample_sigma_square).mean(axis=0)
    
    
    sample_q = []
    sample_n_for_estim_q = []
    print('q')
    for l in range(iters*SAMPLE_SPACING): # On enlève le burn-in
        
        n = compute_n(i)
        
        q = dirichlet.rvs(alpha=q_params+n, random_state=seed)[0]
        
        z,i = compute_conditional_z(q,y,mu_star,sigma_square_star)
        delta = np.apply_along_axis(lambda x : np.sum(x**2), 0, (y*i - np.multiply(i,mu_star)))
        
        sample_q.append(copy.deepcopy(q))
        sample_n_for_estim_q.append(copy.deepcopy(n))
    
    sample_q = np.array(sample_q)[Selected_indexes]
    sample_n_for_estim_q = np.array(sample_n_for_estim_q)[Selected_indexes]

    
    return np.stack(sample_mu), np.stack(sample_sigma_square), np.stack(sample_q), \
            np.stack(sample_mu_hat), np.stack(sample_B), np.stack(sample_n_for_estim_sigma),\
            np.stack(sample_delta), np.stack(sample_n_for_estim_q)
    
    
def compute_marg_likelihood_and_NSE_galaxies(y, iters, init, hypers):
    ''' Compute the marginal likelihood from the Gibbs Sampler output according to Chib (1995)
    y : (array-like) endogeneous variables
    iters: (int) length of the MCMC
    init: (array-like) initialisation parameters
    hypers: (array-like) hyper-parameters
    
    returns: (float) the marginal likelihood/normalizing constant 
    '''
    
    # Initialisation
    d = init['d']
    mu_params, sigma_square_params, q_params = init['mu_params'], init['sigma_square_params'], init['q_params']

    mu, sigma_square, q, mu_hat, B, n_for_estim_sigma, delta, n_for_estim_q = GibbsSampler_galaxies(y, iters, init, hypers)

    mu_star = np.array(mu).mean(axis=0)
    sigma_square_star = np.array(sigma_square).mean(axis=0)
    q_star = np.array(q).mean(axis=0)
    
    ## Marginal likelihood computation P7, right column
    # First term:
    y_given_mu_and_sigma2_stars_pdf = np.stack([norm.pdf(x=y,loc=mu_star[i],scale=sigma_square_star[i]) for i in range(d)])[:,:,0].T
    log_like = np.log((q_star*y_given_mu_and_sigma2_stars_pdf).sum(axis=1)).sum()



    # Second term
    mu_prior = multivariate_normal.logpdf(x=mu_star, mean=mu_params[0], cov=mu_params[1]).sum() # Sum because of a the use of logpdf instead of pdf
    sigma_square_prior = invgamma.logpdf(x=sigma_square_star,a=sigma_square_params[0], scale=np.sqrt(sigma_square_params[1])).sum()
    q_square_prior = dirichlet.logpdf(x=q_star,alpha=q_params).sum()
    
    log_prior = mu_prior+sigma_square_prior+q_square_prior

    # Third term
    conditional_densities_mu = np.array([np.prod(multivariate_normal.pdf(x=mu_star, mean=mu_hat[i], cov=B[i])) for i in range(iters)])
    
    conditional_densities_sigma = np.array([np.prod(invgamma.pdf(x=sigma_square_star, a=(sigma_square_params[0]+n_for_estim_sigma[i])/2,\
                                                         scale=(sigma_square_params[1]+delta[i])/2)) for i in range(iters)])
    
    conditional_densities_q = np.array([dirichlet.pdf(x=q_star, alpha=q_params+n_for_estim_q[i]) for i in range(iters)])

    
    conditional_densities = conditional_densities_mu*conditional_densities_sigma*conditional_densities_q
    
    log_posterior = np.log(conditional_densities.mean())        
    
    log_marg_likelihood = log_like + log_prior - log_posterior

    
    #Numerical Standard Error Computation
    h = np.array([conditional_densities_mu,
                  conditional_densities_sigma,
                  conditional_densities_q])
    
    h_hat = np.array([np.mean(conditional_densities_mu),
                  np.mean(conditional_densities_sigma),
                  np.mean(conditional_densities_q)])
    
    var = compute_var_h_hat(h,h_hat)
    
    NSE = np.dot(np.dot((1/h_hat).reshape(1,-1),var),(1/h_hat).reshape(-1,1))[0,0]
    
    
    return log_marg_likelihood, NSE


#======================================================================================
# Synthetic data simulation
#======================================================================================
def simul_gaussian_mixture(mean, cov, N):
    ''' Simulate a gaussian mixture of means "mean", covariance "cov" and of size n. 
    The number of gaussian in the mixture is implicit and given by the shape of mu/cov
    mean (array-like): The means of the gaussians in the mixture
    cov (ndarray): The (diagonal) covariance matrix of the gaussians 
        (a multivariate gaussian is used rather than d univariate gaussians)
    N: The size of the sample to generate
    -------------------------------------------------------------------------------------
    returns (array-like): The sample generated by the mixture 
    '''
    p = len(mean)
    y = multivariate_normal.rvs(mean=mean, cov=cov, size=int(N/p))
    return np.hstack(y)


def distance_to_true_params_value(G, mu, sigma_square, true_mu_value, true_sigma_value):
    ''' Compute the distance between the true parameters of each gaussian and the estimates of these parameters.
    The parameters are first sorted by mu values in order to compare the estimates closest to the true values of each gaussian
    G: The number of iterations of the Gibbs sampler for each parameter
    mu (ndarray): The estimates of the means of the gaussians at each iteration
    sigma_square (ndarray): The estimates of the variance of the gaussians at each iteration
    true_mu_value (array-like): The actual means of the gaussians
    true_sigma_value (array-like): The actual variance of the gaussians
    ------------------------------------------------------------------------------------
    returns (array-like,array-like): The euclidian distance between the estimates and the true values of the means and the variances at each iteration
    '''
    mu_trajectory = mu.cumsum(axis=0)/np.arange(1, len(mu)+1).reshape(-1,1)
    sigma_square_trajectory = sigma_square.cumsum(axis=0)/np.arange(1, len(sigma_square)+1).reshape(-1,1)

    mu_sorted_trajectory = [] # For each iteration sort the (mu, sigma) components found by mu values
    sigma_square_sorted_trajectory = []

    for i in range(G): # Could be cleaner with a apply along axis
        mu_st, ss_st = (list(t) for t in zip(*sorted(zip(mu_trajectory[i], sigma_square_trajectory[i]))))
        mu_sorted_trajectory.append(mu_st)
        sigma_square_sorted_trajectory.append(ss_st)
    
    true_mu_value, true_sigma_value = (list(t) for t in zip(*sorted(zip(true_mu_value, true_sigma_value))))

    # Compute the distance between the true value and the estimate at each iteration
    mu_dist = cdist(mu_sorted_trajectory, [true_mu_value]).reshape(-1,)
    sigma_square_dist = cdist(sigma_square_sorted_trajectory, [true_sigma_value]).reshape(-1,)
    
    return mu_dist, sigma_square_dist

        