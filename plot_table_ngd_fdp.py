import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd

import gdp

# strongly convex case
def ngd_fdp_sc(
    mu, c_rep, c_seq, t_fdp_seq, t_approxdp_seq, delta_seq, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)

    nc = len(c_seq)
    nt_fdp = len(t_fdp_seq)
    nt_approxdp = len(t_approxdp_seq)
    ndelta = len(delta_seq)
    
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str_fdp = ["Our Bound, t = " + str(t) for t in t_fdp_seq] + ["GDP Composition, t = " + str(t) for t in t_fdp_seq]
    
    fig, axs = plt.subplots(1, 2,
                            figsize = (7, 2))
    fig.tight_layout()
    
    axs[0].xaxis.set_tick_params(labelsize = 6)
    axs[0].yaxis.set_tick_params(labelsize = 6)
    axs[0].set_xticks(np.linspace(0, 1, 3))
    axs[0].set_yticks(np.linspace(0, 1, 3))
    axs[0].set_xlabel("Type I Error",
                      fontsize = 8)
    axs[0].set_ylabel("Type II Error",
                      fontsize = 8)
    
    c = c_rep
    
    # plot f-DP of our bound
    for i in range(nt_fdp):
        t = t_fdp_seq[i]        
        mu0 = np.sqrt((1-c**t) / (1+c**t) * (1 + c) / (1 - c)) * mu
        f_sc = gdp.gdp(mu0, alpha)
        axs[0].plot(alpha, f_sc,
                    color = colors[i],
                    alpha = 0.8) 
    
    # plot f-DP of GDP Composition
    for j in range(nt_fdp):
        t = t_fdp_seq[j]
        mu0 = np.sqrt(t) * mu
        f_original = gdp.gdp(mu0, alpha)
        axs[0].plot(alpha, f_original,
                    color = colors[-(j+1)],
                    linestyle = "--",
                    alpha = 0.8)
    
    axs[0].legend(legend_str_fdp,
                  fontsize = 6)
    
    axs[1].xaxis.set_tick_params(labelsize = 6)
    axs[1].yaxis.set_tick_params(labelsize = 6)
    axs[1].set_xlabel("t",
                      fontsize = 8)
    axs[1].set_ylabel("\u03B5",
                      fontsize = 8)
    
    legend_str_approxdp = ["Our Bound, \u03B4 = {:.0e}".format(delta) for delta in delta_seq] + ["GDP Composition, \u03B4 = {:.0e}".format(delta) for delta in delta_seq]
    
    # plot approximate DP of our bound
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_sc = np.zeros(nt_approxdp)
        
        for i in range(nt_approxdp):
            t = t_approxdp_seq[i]        
            mu_sc = np.sqrt((1-c**t) / (1+c**t) * (1 + c) / (1 - c)) * mu                
            ep_sc[i] = gdp.gdp_to_ep_given_delta(mu_sc, delta)
            
        axs[1].plot(t_approxdp_seq, ep_sc,
                    color = colors[k],
                    marker = "+")

    # plot approximate DP of our GDP Composition          
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_original = np.zeros(nt_approxdp)
        
        for i in range(nt_approxdp):
            t = t_approxdp_seq[i] 
            mu_original = np.sqrt(t) * mu
            ep_original[i] = gdp.gdp_to_ep_given_delta(mu_original, delta)
        
        axs[1].plot(t_approxdp_seq, ep_original,
                    color = colors[-(k+1)],
                    linestyle = "--",
                    marker = "+")
    
    axs[1].legend(legend_str_approxdp,
                  fontsize = 6)
    axs[1].set_xscale("log")
    
    fig.savefig("ngd_fdp_approxdp_sc.pdf",
                format = "pdf",
                bbox_inches = "tight")
    
    nrow = nt_fdp
    ncol = nc + 2
    
    # table of GDP parameters
    array_mu = np.zeros((nrow, ncol))
    
    for i in range(nt_fdp):
        t = t_fdp_seq[i]
        array_mu[i, 0] = t
        for j in range(nc):
            c = c_seq[j]
            array_mu[i, j+1] = np.sqrt((1-c**t) / (1+c**t) * (1 + c) / (1 - c)) * mu
    
        array_mu[i, nc+1] = np.sqrt(t) * mu
    
    df_mu = pd.DataFrame(array_mu)
    df_mu.columns = ["$t \setminus c$"] + [str(c) for c in c_seq] + ["GDP Composition"]
    df_mu.index = [str(t) for t in t_fdp_seq]
    
    s = df_mu.style
    s.format(precision = 3)
    s.format(subset="$t \setminus c$", precision = 0)
    s.hide(axis = "index")
    s.to_latex("ngd_mu_sc.tex",
               encoding = "utf-8",
               position = "H",
               position_float = "centering",
               column_format = ncol * "r",
               hrules = True,
               caption = "GDP parameters over different values of $t$ and $c$, with $L/(n\sigma)={:.1f}$.".format(mu),
               label = "tab:ngd-fdp-sc")
    
# constrained case
# assume D = 1, mu denotes L/n
def ngd_fdp_proj(
    sigma, eta_rep, eta_seq, mu_rep, mu_seq, t_fdp_seq, t_approxdp_seq, delta_seq, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)

    neta = len(eta_seq)
    nmu = len(mu_seq)
    nt_fdp = len(t_fdp_seq)
    nt_approxdp = len(t_approxdp_seq)
    ndelta = len(delta_seq)
    
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str_fdp = ["Our Bound, t \u2265 " + str(t_fdp_seq[0])] + ["GDP Composition, t = " + str(t) for t in t_fdp_seq]
    
    fig, axs = plt.subplots(1, 2,
                            figsize = (7, 2))
    fig.tight_layout()
    
    axs[0].xaxis.set_tick_params(labelsize = 6)
    axs[0].yaxis.set_tick_params(labelsize = 6)
    axs[0].set_xticks(np.linspace(0, 1, 3))
    axs[0].set_yticks(np.linspace(0, 1, 3))
    axs[0].set_xlabel("Type I Error",
                      fontsize = 8)
    axs[0].set_ylabel("Type II Error",
                      fontsize = 8)

    eta = eta_rep
    mu = mu_rep
    
    # plot f-DP of our bound
    mu0 = np.sqrt(3*mu/eta + mu**2 * np.ceil(1/(mu * eta))) / sigma
    f_proj = gdp.gdp(mu0, alpha)
    axs[0].plot(alpha, f_proj,
                color = colors[0],
                alpha = 0.8)     

    # plot f-DP of GDP Composition
    for j in range(nt_fdp):
        t = t_fdp_seq[j]
        f_original = gdp.gdp(mu / sigma * np.sqrt(t), alpha)
        axs[0].plot(alpha, f_original,
                    color = colors[-(j+1)],
                    linestyle = "dashed",
                    alpha = 0.8)

    axs[0].legend(legend_str_fdp,
                  fontsize = 6) 
        
    axs[1].xaxis.set_tick_params(labelsize = 6)
    axs[1].yaxis.set_tick_params(labelsize = 6)
    axs[1].set_xlabel("t",
                      fontsize = 8)
    axs[1].set_ylabel("\u03B5",
                      fontsize = 8)
    
    legend_str_approxdp = ["Our Bound, \u03B4 = {:.0e}".format(delta) for delta in delta_seq] + ["GDP Composition, \u03B4 = {:.0e}".format(delta) for delta in delta_seq]
    
    # plot approximate DP of our bound 
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_proj = np.zeros(nt_approxdp)
        
        for i in range(nt_approxdp):
            t = t_approxdp_seq[i]        
            mu_proj = np.min([mu0, mu / sigma * np.sqrt(t)])              
            ep_proj[i] = gdp.gdp_to_ep_given_delta(mu_proj, delta)
            
        axs[1].plot(t_approxdp_seq, ep_proj,
                    color = colors[k],
                    marker = "+")
    
    # plot approximate DP of our GDP Composition
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_original = np.zeros(nt_approxdp)
        
        for i in range(nt_approxdp):
            t = t_approxdp_seq[i] 
            mu_original = mu / sigma * np.sqrt(t)
            ep_original[i] = gdp.gdp_to_ep_given_delta(mu_original, delta)
        
        axs[1].plot(t_approxdp_seq, ep_original,
                    color = colors[-(k+1)],
                    linestyle = "--",
                    marker = "+")

    axs[1].legend(legend_str_approxdp,
                  fontsize = 6)
    axs[1].set_xscale("log")
    
    fig.savefig("ngd_fdp_approxdp_proj.pdf",
                format = "pdf",
                bbox_inches = "tight")
    
    nrow = nmu
    ncol = neta + 1
    
    # table of thresholded values (E^*, mu^*)
    df_mu = pd.DataFrame(index = range(nrow),
                         columns = ["$L/n \setminus \eta$"] + [str(eta) for eta in eta_seq])

    for i in range(nmu):
        mu = mu_seq[i]
        df_mu.iloc[i, 0] = str(mu)
        for j in range(neta):
            eta = eta_seq[j]

            t_star = np.ceil((3*mu/eta + mu**2 * np.ceil(1/(mu * eta))) / (mu**2))
            mu_star = np.sqrt(3*mu/eta + mu**2 * np.ceil(1/(mu * eta))) / sigma
            df_mu.iloc[i, j+1] = "({:.0f}, {:.3f})".format(t_star, mu_star)
    
    s = df_mu.style
    s.hide(axis = "index")
    s.to_latex("ngd_mu_proj.tex",
               encoding = "utf-8",
               position = "H",
               position_float = "centering",
               column_format = ncol * "r",
               hrules = True,
               caption = "$(t^*, \mu^*)$ over different values of $(L/n, \eta)$, with $\sigma = {:.0f}$.".format(sigma),
               label = "tab:ngd-fdp-proj")

# parameters for strongly convex setting
mu = 0.1
c_rep = 0.99
c_seq = [1 - 0.08, 1 - 0.04, 1 - 0.02, 1 - 0.01, 1 - 0.005]
t_fdp_seq = [10, 100, 1000]
t_approxdp_seq = [10, 20, 40, 80, 160, 320, 640]
delta_seq = [1e-3, 1e-5]

ngd_fdp_sc(mu, c_rep, c_seq, t_fdp_seq, t_approxdp_seq, delta_seq)

# parameters for constrained setting
sigma = 8
eta_rep = 0.1
eta_seq = [0.2, 0.1, 0.05]
mu_rep = 0.5
mu_seq = [0.25, 0.5, 1]
t_fdp_seq = [80, 160, 320]
t_approxdp_seq = [10, 20, 40, 80, 160, 320, 640]
delta_seq = [1e-3, 1e-5]
delta_seq = [1e-3, 1e-5]

ngd_fdp_proj(sigma, eta_rep, eta_seq, mu_rep, mu_seq, t_fdp_seq, t_approxdp_seq, delta_seq)