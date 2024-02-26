import numpy as np
import scipy.optimize

import pandas as pd

import gdp

from prv_accountant import PRVAccountant

from prv_accountant import GaussianMechanism

from prv_SymmPoissonSubsampledGaussianMechanism import SymmPoissonSubsampledGaussianMechanism

# parameters
p = 1500/60000
l = 40
mu = 2/3
delta = 1e-5
E_seq = [50, 100, 200]
c_seq = [0.9999, 0.9998]

# functions for converting RDP to approximate DP
def bbg(
        rho, delta
    ):
        ep = scipy.optimize.minimize(lambda x: rho*x + np.log((x-1)/x) - np.log(delta * x) / (x-1), 2)
        
        return ep.fun

def bs(
    rho, delta
):
    return rho + np.sqrt(4*rho*np.log(1/delta))

def alc(
    rho, delta
):
    ep = scipy.optimize.minimize(lambda x: 1 / (x-1) * np.log((np.exp(rho*(x-1)*x) - 1) / (x * delta) + 1), 2)
    
    return ep.fun
    
def mironov(
    rho, delta
):
    ep = scipy.optimize.minimize(lambda x: rho * x + np.log(1/delta) / (x - 1), 2)
    
    return ep.fun

def rdp_to_approxdp(
    rho, delta
):
    return np.min([bbg(rho, delta),
                   bs(rho, delta),
                   alc(rho, delta),
                   mironov(rho, delta)])


eps_original = []
reg_2 = []
reg_4 = []

multicolumn_mu = pd.MultiIndex.from_arrays([[""] + ["GDP Composition"] * 2 + ["Our Bounds"] * 2,
                                            ["$E \setminus \\text{Algorithms}$", "\\ncgd", "\\nsgd", "$\\ncgd, \lambda = 0.002$", "$\\ncgd, \lambda = 0.004$"]])
multicolumn_ep = pd.MultiIndex.from_arrays([[""] + ["GDP Composition"] * 2 + ["RDP"] * 2 + ["Our Bounds"] * 2,
                                            ["$E \setminus \\text{Algorithms}$", "\\ncgd", "\\nsgd", "$\\ncgd, \lambda = 0.002$", "$\\ncgd, \lambda = 0.004$", "$\\ncgd, \lambda = 0.002$", "$\\ncgd, \lambda = 0.004$"]])

# data frame for GDP parameters
df_mu = pd.DataFrame(columns=multicolumn_mu,
                     index=range(3))
for i in range(len(E_seq)):
    df_mu.iloc[i, 0] = str(E_seq[i])

# data frame for epsilon
df_ep = pd.DataFrame(columns=multicolumn_ep,
                     index=range(3))
for i in range(len(E_seq)):
    df_ep.iloc[i, 0] = str(E_seq[i])

prv_original = SymmPoissonSubsampledGaussianMechanism(sampling_probability=p, mu=mu)

# calculate corresponding GDP parameters and epsilon for each case
for i in range(len(E_seq)):
    E = E_seq[i]
    t = E * l
    
    # value of GDP parameter for GDP Composition on NoisyCGD
    mu0 = mu * np.sqrt(E)
    
    # value of epsilon for GDP Composition on NoisyCGD
    prv_gaussian = GaussianMechanism(noise_multiplier=1/mu0)
    
    accountant = PRVAccountant(
        prvs = [prv_gaussian],
        eps_error=1e-3,
        delta_error=1e-10
    )
    
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[1])
    
    df_mu.iloc[i, 1] = mu0
    df_ep.iloc[i, 1] = eps_est
    
    # value of GDP parameter for GDP Composition on NoisySGD
    mu0 = gdp.clt_mu(p, t, mu)
    
    # value of epsilon for GDP Composition on NoisySGD
    accountant = PRVAccountant(
        prvs=[prv_original],
        max_self_compositions=[t],
        eps_error=1e-3,
        delta_error=1e-10,
        eps_max=30
    )

    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[t])

    eps_original.append(eps_est)
    
    df_mu.iloc[i, 2] = mu0
    
    df_ep.iloc[i, 2] = eps_est
    
    # lambda = 0.002
    c = c_seq[0]
    
    # value of GDP parameter for our bound on NoisyCGD for lambda = 0.002
    mu0 = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu
    
    df_mu.iloc[i, 3] = mu0
    
    # value of epsilon for our bound on NoisyCGD for lambda = 0.002
    prv_gaussian = GaussianMechanism(noise_multiplier=1/mu0)
    
    accountant = PRVAccountant(
        prvs = [prv_gaussian],
        eps_error=1e-3,
        delta_error=1e-10
    )
    
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[1])
        
    reg_2.append(eps_est)
    
    df_ep.iloc[i, 5] = eps_est
    
    # value of rho for RDP bound on NoisyCGD for lambda = 0.002
    rho0 = mu**2 / 2 * (1 + c**(l-2)*(1-c**2)/(1-c**l)**2 * (1-c**(l*(E-1))))
    
    # value of epsilon for RDP bound on NoisyCGD for lambda = 0.004
    df_ep.iloc[i, 3] = rdp_to_approxdp(rho0, delta)
    
    # lambda = 0.004
    c = c_seq[1]

    # value of GDP parameter for our bound on NoisyCGD for lambda = 0.004
    mu0 = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu
    
    df_mu.iloc[i, 4] = mu0
    
    # value of epsilon for our bound on NoisyCGD for lambda = 0.004
    prv_gaussian = GaussianMechanism(noise_multiplier=1/mu0)
    
    accountant = PRVAccountant(
        prvs = [prv_gaussian],
        eps_error=1e-3,
        delta_error=1e-10
    )
    
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[1])
    
    reg_4.append(eps_est)

    df_ep.iloc[i, 6] = eps_est
    
    # value of rho for RDP bound on NoisyCGD for lambda = 0.002
    rho0 = mu**2 / 2 * (1 + c**(l-2)*(1-c**2)/(1-c**l)**2 * (1-c**(l*(E-1))))
    
    # value of epsilon for RDP bound on NoisyCGD for lambda = 0.004
    df_ep.iloc[i, 4] = rdp_to_approxdp(rho0, delta)
    
s = df_mu.style
s.format(precision = 2)
s.hide(axis = "index")
s.to_latex("lr_table_mu.tex",
           encoding = "utf-8",
           position = "H",
           position_float = "centering",
           column_format = 5 * "r",
           multicol_align = "c",
           hrules = True,
           caption = "GDP parameters of private algorithms on regularized logistic regression.",
           label = "tab:lr_mu")

s = df_ep.style
s.format(precision = 2)
s.hide(axis = "index")
s.to_latex("lr_table_ep.tex",
           encoding = "utf-8",
           position = "H",
           position_float = "centering",
           column_format = 7 * "r",
           multicol_align = "c",
           hrules = True,
           caption = "$\ep$ of private algorithms on regularized logistic regression for $\delta = 10^{-5}$.",
           label = "tab:lr_mu")