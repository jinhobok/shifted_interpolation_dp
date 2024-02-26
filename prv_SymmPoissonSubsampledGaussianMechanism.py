import numpy as np

from numpy import exp, log

import scipy.stats

from prv_accountant import PrivacyRandomVariable

class SymmPoissonSubsampledGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, mu: float) -> None:
        self.p = np.longdouble(sampling_probability)
        self.mu = np.longdouble(mu)
    
    def cdf(self, t):
        p = self.p
        mu = self.mu
        
        return np.where(t > 0,
                        p * scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu - mu/2) + (1-p)*scipy.stats.norm.cdf(log((p-1 + exp(t))/p)/mu + mu/2),
                        scipy.stats.norm.cdf(-log((p-1 + exp(-t))/p)/mu - mu/2))