import numpy as np
import scipy.optimize
import scipy.stats

from prv_accountant import PRVAccountant

from prv_accountant import GaussianMechanism

# calculate G(mu) at alpha
def gdp(
    mu, alpha
):
    z = scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - alpha) - mu)
    return z

# calculate C_p(G(mu))^t at alpha
def gdp_clt(
    p, t, mu, alpha
):
    p0 = p * np.sqrt(t)
    mu0 = np.sqrt(2) * p0 * np.sqrt(np.exp(mu**2) * scipy.stats.norm.cdf(1.5 * mu) + 3 * scipy.stats.norm.cdf(-0.5 * mu) - 2)
    return gdp(mu0, alpha)

# calculate approximation (by CLT) for the strongly convex case given a grid (grid_seq = t - tau) at alpha
def gdp_clt_sc(
    p, t, c, mu, alpha, mode, grid_seq
):
    if mode == "gridsearch":
        nu = grid_seq

        p0 = p * np.sqrt(nu + 1)
        mu_clt = 2 * mu
        mu0 = np.min(np.sqrt(2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2 + 2 * (mu_clt * (c**(nu + 1) - c**t) / (1 - c))**2))
        
        return gdp(mu0, alpha)

# calculate approximation (by CLT) for the constrained case given a grid (grid_seq = t - tau) at alpha
# Assume D = 1, mu denotes L/b
def gdp_clt_proj(
    p, t, sigma, eta, mu, alpha, mode, grid_seq
):
    if mode == "gridsearch":
        tau = grid_seq
        
        p0 = p * np.sqrt(t - tau)
        mu_clt = 2*np.sqrt(2)*mu / sigma
        mu0 = np.min(np.sqrt(2*(1/(eta * sigma))**2 / (t - tau) + 2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2))
        
        return gdp(mu0, alpha)

# given mu and delta, return converted epsilon for G(mu)    
def gdp_to_ep_given_delta(
    mu, delta
):
    
    prv_gaussian = GaussianMechanism(noise_multiplier=1/mu)
    
    accountant = PRVAccountant(
        prvs = [prv_gaussian],
        eps_error=1e-3,
        delta_error=1e-10
    )
    
    eps_low, ep, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[1])
      
    return ep

# given mu and epsilon, return converted delta for G(mu)    
def gdp_to_delta_given_ep(
    mu, ep    
):
    z = scipy.stats.norm.cdf(-ep / mu + 0.5 * mu) -np.exp(ep)*scipy.stats.norm.cdf(-ep/mu - 0.5 *mu)
    return z

# calculate corresponding GDP parameter (given by CLT) for subsampled Gaussian mechanism
def clt_mu(
    p, t, mu
):
    p0 = p * np.sqrt(t)
    return np.sqrt(2) * p0 * np.sqrt(np.exp(mu**2)*scipy.stats.norm.cdf(1.5 * mu) + 3 * scipy.stats.norm.cdf(-0.5 * mu) - 2)

# calculate corresponding GDP parameter (given by CLT) for one-sided subsampled Gaussian mechanism (see Bu et al., 2020)
def clt_mu_onesided(
    p, t, mu
):
    p0 = p * np.sqrt(t)
    return p0 * np.sqrt(np.exp(mu**2) - 1)

# calculate the tradeoff function corresponding to (ep, delta)-DP at alpha
def ep_delta_to_fdp(
    ep, delta, alpha
):
    thres = (1-delta)/(1 + np.exp(ep))
    return np.where(alpha < thres,
                    1 - delta - np.exp(ep)*alpha,
                    np.where(alpha < 1-delta,
                             np.exp(-ep)*(1 - delta - alpha),
                             0))
    