import numpy as np
from .py3 import *

# Monotonic non increasing function
def pav (y):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y

    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    Parameters
    ----------
    y: list
    
    Returns
    -------
    v: list
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv <= 0):
            break

        viol = np.where(deriv > 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v


###############################################################################
# analytical solutions to the equations in Walker 2009
# use these in the EM algorithm to construct the method


def mean (sig, p_m, vec, err_vec, model=None):
    """
    Walker 2009 equation 13
    
    sig is passed in differently every time. Used to iterate and find the mean
    value of the data. 

    Parameters
    ----------
    sig: float
    p_m: np.array
    vec: np.array
    err_vec: np.array
    model: np.array
    
    Returns
    -------
    mean: float
    """
    
    if model == None:
        model = np.ones(len(vec))

    divisor = (1.+ (err_vec**2/(sig*model)**2))
    numerator = (p_m*vec)/divisor
    denominator = (p_m)/divisor
    return np.sum(numerator)/np.sum(denominator)  

def variance (mean, sig, p_m, vec, err_vec, model=None):
    """
    Walker 2009 equation 14
    
    mean and sig are passed in differently every time.
    
    Parameters
    ----------
    mean: float
    sig: float
    p_m: np.array
    vec: np.array
    err_vec: np.array
    model: np.array
    
    Returns
    -------
    variance: float
    """
    if model == None:
        model = np.ones(len(vec))

    divisor = (1.0 + (err_vec**2/(sig*model)**2))
    numerator = (p_m*(vec - mean)**2)/divisor**2
    denominator = (p_m*model**2)/divisor
    return np.sum(numerator)/np.sum(denominator)

def mean_non (sig, p_m, vec, err_vec, model=None):
    """
    Walker 2009 equation 13
    
    sig is passed in differently every time. Used to iterate and find the mean
    value of the data. 

    Parameters
    ----------
    sig: float
    p_m: np.array
    vec: np.array
    err_vec: np.array
    model: np.array
    
    Returns
    -------
    mean: float
    """
    
    if model == None:
        model = np.ones(len(vec))

    divisor = (1. + (err_ec**2/sig**2))
    numerator = ((1.0 - p_m)*vec)/divisor
    denominator = (1.0 - p_m)/divisor
    return np.sum(numerator)/np.sum(denominator)  

def variance_non (mean, sig, p_m, vec, err_vec, model=None):
    """
    Walker 2009 equation 14
    
    mean and sig are passed in differently every time.
    
    Parameters
    ----------
    mean: float
    sig: float
    p_m: np.array
    vec: np.array
    err_vec: np.array
    model: np.array
    
    Returns
    -------
    variance: float
    """
    if model == None:
        model = np.ones(len(vec))

    divisor = (1.0 + (err_vec**2/(sig*model)**2))
    numerator = ((1.0 - p_m)*(vec - mean)**2)/divisor**2
    denominator = ((1.0 - p_m)*model**2)/divisor
    return np.sum(numerator)/np.sum(denominator)


###############################################################################
# probability distributions 1d

def p_normalized (sig, mean, vec, err_vec, model=None):
    """
    Walker 2009 equation 9
    
    vbar and sig0 are passed in differently every time.
    
    Parameters
    ----------
    sig: float
    mean: float
    vec: np.array
    err_vec: np.array
    model: np.array
    
    Returns
    -------
    membership: np.array
    """ 
    if model == None:
        model = np.ones(len(vec))
           
    two_pi = np.pi*2.
    
    v_sig = ((sig*model)**2 + err_vec**2)
    norm = 1.0/(np.sqrt(two_pi*v_sig))
    v_ = ((vec - mean)**2/((sig*model)**2 + err_vec**2))
    expon = np.exp(-0.5*(v_))
    return norm*expon

def _p_contamination (vec_i, contamination_model):
    """
    Walker 2009 equation 7
    Parameters
    ----------
    vec_i: float
    
    Returns
    -------
    contamination_probaility: float
    """        
    sig_model = 20.0 # 20 for your paper

    n_model = len(contamination_model)

    norm = 1.0/np.sqrt(2.*np.pi*sig_model**2)
    expon = np.exp((-0.5*(contamination_model - vec_i)**2)/sig_model**2)
    over_N = (1.0/n_model)
    return over_N*np.sum(norm*expon) 

def p_contamination_non (vec, contamination_model): 
    """
    Walker 2009 equation 10
    
    Parameters
    ----------
    vec: np.array
    contamination_model: np.array
    
    Returns
    -------
    non_member_probabilities: np.array
    """            
    p_model = []
    for i in xrange(len(vec)):
        P_ = _p_contamination(vec[i], contamination_model)
        p_model.append(P_)
        
    return np.array(p_model)



###############################################################################
# likelihood equations

def normalized_probs (p_mem, p_non, p_a):
    """
    Walker 2009 equation 11
    Parameters
    ----------
    p_mem: np.array
    p_non: np.array
    p_a: np.array
    
    Returns
    -------
    norm_probs: np.array
    """
    
    p_m = (p_mem*p_a)/(p_mem*p_a + p_non*(1.0 - p_a))
    return p_m 


def neg_log_likelihood (p_m, p_a, p_mem, p_non):
    """
    negative log likelihood function
    
    Parameters
    ----------
    p_m: np.array
    p_a: np.array
    p_mem: np.array
    p_non: np.array
    
    Returns
    -------
    neg_log_like: float
    """
    mem = p_mem*p_a
    non = p_non*(1.0-p_a)
    log_like_term1 = np.sum(p_m*np.log(np.where(mem != 0, mem, 1)))
    log_like_term2 = np.sum((1.0-p_m)*np.log(np.where(non !=0, non, 1)))
    return -(log_like_term1 + log_like_term2)

def log_likelihood (p_m, p_a, p_mem, p_non):
    """
    log likelihood function
    
    Parameters
    ----------
    p_m: np.array
    p_a: np.array
    p_mem: np.array
    p_non: np.array
    
    Returns
    -------
    log_like: float
    """
    mem = p_mem*p_a
    non = p_non*(1.0-p_a)
    log_like_term1 = np.sum(p_m*np.log(np.where(mem != 0, mem, 1)))
    log_like_term2 = np.sum((1.0-p_m)*np.log(np.where(non !=0, non, 1)))
    return log_like_term1 + log_like_term2 


    
