import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd

import gdp

# strongly convex case
def ncgd_fdp_sc(
    mu, c_rep, c_seq, l_rep, l_seq, E_fdp_seq, E_approxdp_seq, delta_seq, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)

    nc = len(c_seq)
    nl = len(l_seq)
    nE_fdp = len(E_fdp_seq)
    nE_approxdp = len(E_approxdp_seq)
    ndelta = len(delta_seq)
    
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str_fdp = ["Our Bound, E = " + str(E) for E in E_fdp_seq] + ["GDP Composition, E = " + str(E) for E in E_fdp_seq]
    
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
    l = l_rep
    
    # plot f-DP of our bound
    for i in range(nE_fdp):
        E = E_fdp_seq[i]        
        mu0 = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu
        f_sc = gdp.gdp(mu0, alpha)
        axs[0].plot(alpha, f_sc,
                    color = colors[i],
                    alpha = 0.8) 
    
    # plot f-DP of GDP Composition
    for j in range(nE_fdp):
        E = E_fdp_seq[j]
        mu0 = np.sqrt(E) * mu
        f_original = gdp.gdp(mu0, alpha)
        axs[0].plot(alpha, f_original,
                    color = colors[-(j+1)],
                    linestyle = "--",
                    alpha = 0.8)
    
    axs[0].legend(legend_str_fdp,
                  fontsize = 6)
    
    axs[1].xaxis.set_tick_params(labelsize = 6)
    axs[1].yaxis.set_tick_params(labelsize = 6)
    axs[1].set_xlabel("E",
                      fontsize = 8)
    axs[1].set_ylabel("\u03B5",
                      fontsize = 8)
    
    legend_str_approxdp = ["Our Bound, \u03B4 = {:.0e}".format(delta) for delta in delta_seq] + ["GDP Composition, \u03B4 = {:.0e}".format(delta) for delta in delta_seq]
    
    # plot approximate DP of our bound
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_sc = np.zeros(nE_approxdp)
        
        for i in range(nE_approxdp):
            E = E_approxdp_seq[i]        
            mu_sc = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu              
            ep_sc[i] = gdp.gdp_to_ep_given_delta(mu_sc, delta)
            
        axs[1].plot(E_approxdp_seq, ep_sc,
                    color = colors[k],
                    marker = "+")

    # plot approximate DP of our GDP Composition            
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_original = np.zeros(nE_approxdp)
        
        for i in range(nE_approxdp):
            E = E_approxdp_seq[i] 
            mu_original = np.sqrt(E) * mu
            ep_original[i] = gdp.gdp_to_ep_given_delta(mu_original, delta)
        
        axs[1].plot(E_approxdp_seq, ep_original,
                    color = colors[-(k+1)],
                    linestyle = "--",
                    marker = "+")
    
    axs[1].legend(legend_str_approxdp,
                  fontsize = 6)
    axs[1].set_xscale("log")
    
    fig.savefig("ncgd_fdp_approxdp_sc.pdf",
                format = "pdf",
                bbox_inches = "tight")
    
    nrow = nE_fdp
    ncol = nc*nl + 2
    
    # table of GDP parameters
    df_mu = pd.DataFrame(index = range(nrow),
                         columns = range(ncol))

    for i in range(nE_fdp):
        E = E_fdp_seq[i]
        df_mu.iloc[i, 0] = str(E)
    
        for k in range(nl):
            l = l_seq[k]
        
            for j in range(nc):
                c = c_seq[j]
                df_mu.iloc[i, nl*k + j+1] = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu
        
        df_mu.iloc[i, nc*nl + 1] = np.sqrt(E) * mu
    
    multicolumn = pd.MultiIndex.from_arrays([[""] + [item for col in [["$l = " + str(l) + "$"] * nc for l in l_seq] for item in col] + [""],
                                            ["$E \setminus c$"] + [str(c) for c in c_seq] * nl + ["GDP Composition"]])
    df_mu.columns = multicolumn
    
    s = df_mu.style
    s.format(precision = 3)
    s.hide(axis = "index")
    s.to_latex("ncgd_mu_sc.tex",
               encoding = "utf-8",
               position = "H",
               position_float = "centering",
               column_format = ncol * "r",
               multicol_align = "c",
               hrules = True,
               caption = "GDP parameters over different values of $E, l$ and $c$, with $L/(b\sigma)={:.1f}$.".format(mu),
               label = "tab:ncgd-fdp-sc")
    
# constrained case
# assume D = 1, mu denotes L/n
def ncgd_fdp_proj(
    sigma, eta_rep, eta_seq, mu_rep, mu_seq, l_rep, l_seq, E_fdp_seq, E_approxdp_seq, delta_seq, num_col = 10
):
    
    alpha = np.linspace(0, 1, 500)

    neta = len(eta_seq)
    nmu = len(mu_seq)
    nl = len(l_seq)
    nE_fdp = len(E_fdp_seq)
    nE_approxdp = len(E_approxdp_seq)
    ndelta = len(delta_seq)
    
    colors = cm.get_cmap("turbo")
    colors = colors(np.linspace(0.05, 0.95, num_col))
    
    legend_str_fdp = ["Our Bound, E \u2265 " + str(E_fdp_seq[0])] + ["GDP Composition, E = " + str(E) for E in E_fdp_seq]
    
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
    l = l_rep

    # plot f-DP of our bound
    mu0 = np.sqrt(3*mu/(eta * l) + mu**2 + mu**2/l * np.ceil(1 / (mu * eta))) / sigma
    f_proj = gdp.gdp(mu0, alpha)
    axs[0].plot(alpha, f_proj,
                color = colors[0],
                alpha = 0.8)     

    # plot f-DP of GDP Composition    
    for j in range(nE_fdp):
        E = E_fdp_seq[j]
        f_original = gdp.gdp(mu / sigma * np.sqrt(E), alpha)
        axs[0].plot(alpha, f_original,
                    color = colors[-(j+1)],
                    linestyle = "dashed",
                    alpha = 0.8)

    axs[0].legend(legend_str_fdp,
                  fontsize = 6) 
        
    axs[1].xaxis.set_tick_params(labelsize = 6)
    axs[1].yaxis.set_tick_params(labelsize = 6)
    axs[1].set_xlabel("E",
                      fontsize = 8)
    axs[1].set_ylabel("\u03B5",
                      fontsize = 8)
    
    legend_str_approxdp = ["Our Bound, \u03B4 = {:.0e}".format(delta) for delta in delta_seq] + ["GDP Composition, \u03B4 = {:.0e}".format(delta) for delta in delta_seq]

    # plot approximate DP of our bound    
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_proj = np.zeros(nE_approxdp)
        
        for i in range(nE_approxdp):
            E = E_approxdp_seq[i]        
            mu_proj = np.min([mu0, mu / sigma * np.sqrt(E)])              
            ep_proj[i] = gdp.gdp_to_ep_given_delta(mu_proj, delta)
            
        axs[1].plot(E_approxdp_seq, ep_proj,
                    color = colors[k],
                    marker = "+")
        
    # plot approximate DP of our GDP Composition    
    for k in range(ndelta):
        delta = delta_seq[k]
        ep_original = np.zeros(nE_approxdp)
        
        for i in range(nE_approxdp):
            E = E_approxdp_seq[i] 
            mu_original = mu / sigma * np.sqrt(E)
            ep_original[i] = gdp.gdp_to_ep_given_delta(mu_original, delta)
        
        axs[1].plot(E_approxdp_seq, ep_original,
                    color = colors[-(k+1)],
                    linestyle = "--",
                    marker = "+")
    
    axs[1].legend(legend_str_approxdp,
                  fontsize = 6)
    axs[1].set_xscale("log")
    
    fig.savefig("ncgd_fdp_approxdp_proj.pdf",
                format = "pdf",
                bbox_inches = "tight")
    
    nrow = nmu
    ncol = neta + 1
    
    # table of thresholded values (E^*, mu^*)
    for k in range(nl):
        df_mu = pd.DataFrame(index = range(nrow),
                             columns = ["$L/b \setminus \eta$"] + [str(eta) for eta in eta_seq])

        l = l_seq[k]
    
        for i in range(nmu):
            mu = mu_seq[i]
            df_mu.iloc[i, 0] = str(mu)
            
            for j in range(neta):
                eta = eta_seq[j]
        
                E_star = np.ceil((3*mu/(eta*l) + mu**2 * mu**2 / l * np.ceil(1/(mu * eta))) / (mu**2))
                mu_star = np.sqrt(3*mu/(eta * l) + mu**2 + mu**2/l * np.ceil(1 / (mu * eta))) / sigma
        
                df_mu.iloc[i, j+1] = "({:.0f}, {:.3f})".format(E_star, mu_star)
            
    
        s = df_mu.style
        s.hide(axis = "index")
        s.to_latex("ncgd_mu_proj_{:.0f}.tex".format(l),
                   encoding = "utf-8",
                   position = "H",
                   position_float = "centering",
                   column_format = ncol * "r",
                   hrules = True,
                   caption = "$(E^*, \mu^*)$ over different values of $(L/b, \eta)$, with $l = {:.0f}$ and $\sigma = {:.0f}$.".format(l, sigma),
                   label = "tab:ngd-fdp-proj-{:.0f}".format(l))

# parameters for strongly convex setting
mu = 0.2
c_rep = 0.99
c_seq = [1-0.02, 1-0.01, 1-0.005]
l_rep = 20
l_seq = [10, 20, 40]
E_fdp_seq = [5, 50, 500]
E_approxdp_seq = [5, 10, 20, 40, 80, 160, 320]
delta_seq = [1e-3, 1e-5]

ncgd_fdp_sc(mu, c_rep, c_seq, l_rep, l_seq, E_fdp_seq, E_approxdp_seq, delta_seq)

# parameters for constrained setting
sigma = 3
eta_rep = 0.02
eta_seq = [0.04, 0.02, 0.01]
mu_rep = 0.5
mu_seq = [0.25, 0.5, 1]
l_rep = 20
l_seq = [10, 20, 40]
E_fdp_seq = [25, 50, 100]
E_approxdp_seq = [5, 10, 20, 40, 80, 160, 320]
delta_seq = [1e-3, 1e-5]

ncgd_fdp_proj(sigma, eta_rep, eta_seq, mu_rep, mu_seq, l_rep, l_seq, E_fdp_seq, E_approxdp_seq, delta_seq)