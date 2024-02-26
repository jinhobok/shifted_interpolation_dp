import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import gdp

import scipy.optimize

# color palette
colors = cm.get_cmap("turbo")
colors = colors(np.linspace(0.05, 0.95, 10))

# plot of f-DP, GDP example
def piecewise_linear(
    pivot, alpha
):
    return np.where(alpha <= pivot,
                    -(1-pivot)/pivot * alpha + 1,
                    pivot/(1-pivot)*(1 - alpha))

alpha = np.linspace(0, 1, 500)

G_half = gdp.gdp(0.5, alpha)
G_1 = gdp.gdp(1, alpha)
f = piecewise_linear(0.37, alpha)
G_5 = gdp.gdp(5, alpha)

plt.plot(alpha, 1-alpha,
         color = "black")
plt.plot(alpha, G_half,
         color = colors[2])
plt.plot(alpha, G_1,
         color = colors[3])
plt.plot(alpha, G_5,
         color = colors[-1])
plt.plot(alpha, f,
         color = "black",
         linestyle = "--")
plt.legend(["G(0) = Id (full privacy)",
            "G(0.5)",
            "G(1)",
            "G(5) (essentially no privacy)",
            "T$(\mathcal{A}(S), \mathcal{A}(S'))$"],
            bbox_to_anchor = (1, 0.75))
plt.xticks(np.linspace(0, 1, 3))
plt.yticks(np.linspace(0, 1, 3))
plt.gcf().set_size_inches(2, 2)
plt.savefig("fdp_intro.pdf",
            format = "pdf",
            bbox_inches = "tight")
plt.clf()

fig, axs = plt.subplots(1, 2,
                        figsize = (4, 2),
                        layout = "constrained")
fig.tight_layout()

# =============================================================================
# plot of toy example of strongly convex, NoisyGD in the main text
alpha = np.linspace(0, 1, 500)

delta_seq = [1e-5]
t_approxdp_seq = [10, 20, 40, 80, 160]
c = 0.95
c_ys = 0.975
mu = 1/10

ndelta = len(delta_seq)
nt_approxdp = len(t_approxdp_seq)

# formulae for converting RDP to approximate DP
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

axs[0].plot(alpha, 1-alpha,
            color = "black",
            linestyle = ":")

for k in range(ndelta):
    delta = delta_seq[k]
    ep_sc = np.zeros(nt_approxdp)
    
    for i in range(nt_approxdp):
        t = t_approxdp_seq[i]        
        mu_sc = np.sqrt((1-c**t) / (1+c**t) * (1 + c) / (1 - c)) * mu                
        ep_sc[i] = gdp.gdp_to_ep_given_delta(mu_sc, delta)
        
        if i == 4:
            f_sc = gdp.gdp(mu_sc, alpha)
            axs[0].plot(alpha, f_sc,
                        color = colors[0])
        
    axs[1].plot(t_approxdp_seq, ep_sc,
             color = colors[0],
             marker = "+")

for k in range(ndelta):
    delta = delta_seq[k]
    ep_renyiopt = np.zeros(nt_approxdp)
    
    for i in range(nt_approxdp):
        t = t_approxdp_seq[i] 
        mu_sc = np.sqrt((1-c**t) / (1+c**t) * (1 + c) / (1 - c)) * mu
        
        ep_bbg = bbg(mu_sc**2 / 2, delta)
        ep_bs = bs(mu_sc**2 / 2, delta)
        ep_alc = alc(mu_sc**2 / 2, delta)
        ep_mironov = mironov(mu_sc**2 / 2, delta)
        
        ep_renyiopt[i] = np.min([ep_bbg, ep_bs, ep_alc, ep_mironov])
    
    plt.plot(t_approxdp_seq, ep_renyiopt,
             color = colors[2],
             marker = "+")

for k in range(ndelta):
    delta = delta_seq[k]
    ep_ys = np.zeros(nt_approxdp)
    
    for i in range(nt_approxdp):
        t = t_approxdp_seq[i] 
        mu_ys = np.sqrt(2*(1+c)/(1-c) * (1-c_ys**t)) * mu
        
        ep_bbg = bbg(mu_ys**2 / 2, delta)
        ep_bs = bs(mu_ys**2 / 2, delta)
        ep_alc = alc(mu_ys**2 / 2, delta)
        ep_mironov = mironov(mu_ys**2 / 2, delta)
        
        ep_ys[i] = np.min([ep_bbg, ep_bs, ep_alc, ep_mironov])
            

    axs[1].plot(t_approxdp_seq, ep_ys,
             color = colors[3],
             marker = "+")

for k in range(ndelta):
    delta = delta_seq[k]
    ep_original = np.zeros(nt_approxdp)
    
    for i in range(nt_approxdp):
        t = t_approxdp_seq[i] 
        mu_original = np.sqrt(t) * mu
        ep_original[i] = gdp.gdp_to_ep_given_delta(mu_original, delta)
    
        if i == 4:
            f_original = gdp.gdp(mu_original, alpha)
            axs[0].plot(alpha, f_original,
                        color = colors[-3])
    
    axs[1].plot(t_approxdp_seq, ep_original,
             color = colors[-3],
             marker = "+")

f_noprivacy = gdp.gdp(10, alpha)
axs[0].plot(alpha, f_noprivacy,
            color = colors[-1],
            linestyle = ":")

axs[0].legend(["Perfect privacy",
               "Our $f$-DP bound\n(optimal)",
               "Composition bound",
               "No privacy"],
              fontsize = 4.5)
axs[0].xaxis.set_tick_params(labelsize = 6)
axs[0].yaxis.set_tick_params(labelsize = 6)
axs[0].set_xticks(np.linspace(0, 1, 3))
axs[0].set_yticks(np.linspace(0, 1, 3))
axs[0].set_xlabel("Type I Error",
                  fontsize = 8)
axs[0].set_ylabel("Type II Error",
                  fontsize = 8)

axs[1].set_xlabel("Iterations t",
                  fontsize = 8)
axs[1].set_ylabel("Privacy \u03B5",
                  fontsize = 8)
axs[1].xaxis.set_tick_params(labelsize = 6)
axs[1].yaxis.set_tick_params(labelsize = 6)
axs[1].legend(["Our $f$-DP bound\n(optimal)",
               "Our RDP bound",
               "Prior RDP bound",
               "Composition bound"],
              fontsize = 4.5)
axs[1].set_xscale("log")


fig.savefig("fdp_opt.pdf",
            format = "pdf",
            bbox_inches = "tight")