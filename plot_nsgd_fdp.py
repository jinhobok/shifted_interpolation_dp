import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm

import gdp

# strongly convex case
def nsgd_fdp_sc(
    p_seq, mu_seq, c_seq, t_seq, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)
    nrow = len(p_seq)
    ncol = len(mu_seq)
    nc = len(c_seq)
    len_t = len(t_seq)
            
    # color palette
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str = ["c = " + str(c) for c in c_seq] + ["GDP Composition, t = " + str(t_seq[j]) for j in range(len_t)]

    fig, axs = plt.subplots(nrow, ncol,
                            figsize = (7, 5))
    fig.tight_layout()
    for row in range(nrow):
        p = p_seq[row]
        for column in range(ncol):
            mu = mu_seq[column]
            axs[row, column].xaxis.set_tick_params(labelsize = 6)
            axs[row, column].yaxis.set_tick_params(labelsize = 6)
            axs[row, column].set_xticks(np.linspace(0, 1, 3))
            axs[row, column].set_yticks(np.linspace(0, 1, 3))
            
            # plot f-DP of our bound 
            for i in range(nc):
                c = c_seq[i]
                # grid of nu = t - tau
                nu = np.arange(100, np.min(t_seq), 100)
                p0 = p * np.sqrt(nu + 1)
                mu_clt = 2 * mu
                mu0 = np.min(np.sqrt(2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2 + 2 * (mu_clt * (c**(nu + 1)) / (1 - c))**2))
                
                f_sc = gdp.gdp(mu0, alpha)
                axs[row, column].plot(alpha, f_sc,
                                      color = colors[i],
                                      alpha = 0.8)  
            
            # plot f-DP of GDP Composition
            for j in range(len_t): 
                t = t_seq[j]
                f_original = gdp.gdp_clt(p, t, mu, alpha)
                axs[row, column].plot(alpha, f_original,
                                      color = colors[-(j+1)],
                                      linestyle = "dashed",
                                      alpha = 0.8)  
            
            axs[row, column].set_title("b/n = " + str(p) + ", L/(b\u03C3) = " + str(mu), fontsize = 6)
    
    plt.figlegend(legend_str,
                    loc = "lower center",
                    ncol = 2,
                    bbox_to_anchor = (0.5, -0.1),
                    fontsize = 6)
    fig.savefig("nsgd_fdp_sc.pdf" ,
                format = "pdf",
                bbox_inches = "tight")

# constrained case
# assume D = 1, mu denotes L/n
def nsgd_fdp_proj(
    p_seq, eta_seq, mu_seq, t_seq, sigma, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)
    nrow = len(p_seq)
    ncol = len(mu_seq)
    neta = len(eta_seq)
    len_t = len(t_seq)
    
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str = ["\u03B7 = " + str(eta_seq[i]) for i in range(neta)] + ["GDP Composition, t = " + str(t_seq[j]) for j in range(len_t)]

    fig, axs = plt.subplots(nrow, ncol,
                            figsize = (7, 5))
    fig.tight_layout()
    for row in range(nrow):
        p = p_seq[row]
        for column in range(ncol):
            mu = mu_seq[column]
            axs[row, column].xaxis.set_tick_params(labelsize = 6)
            axs[row, column].yaxis.set_tick_params(labelsize = 6)
            axs[row, column].set_xticks(np.linspace(0, 1, 3))
            axs[row, column].set_yticks(np.linspace(0, 1, 3))
            
            # plot f-DP of our bound 
            for i in range(neta):
                eta = eta_seq[i]
                # grid of nu = t - tau
                nu = np.arange(100, np.min(t_seq), 100)
                p0 = p * np.sqrt(nu)
                mu_clt = 2*np.sqrt(10/9)*mu / sigma
                mu0 = np.min(np.sqrt(10*(1/(eta * sigma))**2 / nu + 2 * (np.exp(mu_clt**2) * scipy.stats.norm.cdf(1.5 * mu_clt) + 3 * scipy.stats.norm.cdf(-0.5 * mu_clt) - 2) * p0**2))
                
                f_proj = gdp.gdp(mu0, alpha)
                axs[row, column].plot(alpha, f_proj,
                                      color = colors[i],
                                      alpha = 0.8) 
            
            # plot f-DP of GDP Composition
            for j in range(len_t):
                t = t_seq[j] 
                f_original = gdp.gdp_clt(p, t, mu / sigma, alpha)
                axs[row, column].plot(alpha, f_original,
                                      color = colors[-(j+1)],
                                      linestyle = "dashed",
                                      alpha = 0.8)  
            
            axs[row, column].set_title("b/n = " + str(p) + ", L/b = " + str(mu), fontsize = 6)
        
        plt.figlegend(legend_str,
                      loc = "lower center",
                      ncol = 2,
                      bbox_to_anchor = (0.5, -0.1),
                      fontsize = 6)
        fig.savefig("nsgd_fdp_proj.pdf" ,
                    format = "pdf",
                    bbox_inches = "tight") 

# parameters for strongly convex setting        
p_seq = [0.0025, 0.005, 0.01, 0.02, 0.04]
mu_seq = [0.05, 0.1, 0.2, 0.4, 0.8]
c_seq = [1 - 0.02, 1-0.01, 1-0.005]
t_seq = [5000, 20000, 80000]

nsgd_fdp_sc(p_seq, mu_seq, c_seq, t_seq)

# parameters for constrained setting
p_seq = [0.0025, 0.005, 0.01, 0.02, 0.04]
eta_seq = [0.2, 0.1, 0.05]
mu_seq = [0.125, 0.25, 0.5, 1, 2]
t_seq = [5000, 20000, 80000]
sigma = 3

nsgd_fdp_proj(p_seq, eta_seq, mu_seq, t_seq, sigma)