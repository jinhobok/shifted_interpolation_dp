import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm

# color palette
colors = cm.get_cmap("turbo")
colors = colors(np.linspace(0.05, 0.95, 10))

# plot for illustrating C_p(G(mu))
p = 0.25
mu = 2.5
alpha = np.linspace(0, 1, 500)
f1 = p * scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - alpha) - mu) + (1-p)*(1-alpha)

alpha_x = scipy.stats.norm.cdf(-0.5*mu)
alpha_y = p * scipy.stats.norm.cdf(-0.5*mu) + (1-p)*scipy.stats.norm.cdf(0.5 * mu)

alpha_cp1 = np.linspace(0, alpha_x, 500)
alpha_cp2 = np.linspace(alpha_x, alpha_y, 500)

f_cp1 = p * scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - alpha_cp1) - mu) + (1-p) * (1-alpha_cp1)
f_cp2 = (1 + p) * scipy.stats.norm.cdf(-0.5*mu) + (1-p) * scipy.stats.norm.cdf(0.5 * mu) - alpha_cp2

z = np.linspace(-10, 10, 500)
alpha_0 = 1 - (1-p)*scipy.stats.norm.cdf(z) - p*scipy.stats.norm.cdf(z + mu)
alpha_50 = 1 - (1-p)*scipy.stats.norm.cdf(z) - p*scipy.stats.norm.cdf(z + 0.5 * mu)
alpha_100 = 1 - scipy.stats.norm.cdf(z)

f_0 = scipy.stats.norm.cdf(z) 
f_50 = (1-p) * scipy.stats.norm.cdf(z) + p * scipy.stats.norm.cdf(z - 0.5 * mu)
f_100 = (1-p) * scipy.stats.norm.cdf(z) + p * scipy.stats.norm.cdf(z - mu)

plt.plot(alpha_0, f_0,
         color = colors[0],
         alpha = 0.8,
         label = "$f^{(0)}$")
plt.plot(alpha_50, f_50,
         color = colors[2],
         alpha = 0.8,
         label = "$f^{(0.5)}$")
plt.plot(alpha_100, f_100,
         color = colors[4],
         alpha = 0.8,
         label = "$f^{(1)}$")
plt.plot(alpha_cp1, f_cp1,
         color = "black",
         alpha = 0.8)
plt.plot(alpha_cp2, f_cp2,
         color = "black",
         alpha = 0.8)
plt.plot(f_cp1, alpha_cp1,
         color = "black",
         alpha = 0.8,
         label = "$C_p(f)$")
plt.plot(p * scipy.stats.norm.cdf(-0.5 * mu) + (1-p)*scipy.stats.norm.cdf(0.5*mu),
         scipy.stats.norm.cdf(-0.5 * mu),
         "o",
         color = colors[0],
         ms = 5)
plt.plot(p * scipy.stats.norm.cdf(-0.5 * mu) + (1-p)*0.5,
         p * scipy.stats.norm.cdf(-0.5 * mu) + (1-p)*0.5,
         "o",
         color = colors[2],
         ms = 5)
plt.plot(scipy.stats.norm.cdf(-0.5 * mu),
         p * scipy.stats.norm.cdf(-0.5 * mu) + (1-p)*scipy.stats.norm.cdf(0.5*mu),
         "o",
         color = colors[4],
         ms = 5)
plt.xticks(np.linspace(0, 1, 3))
plt.yticks(np.linspace(0, 1, 3))
plt.legend()
plt.gcf().set_size_inches(3, 3)
plt.savefig("nsgd_opt.pdf",
            format = "pdf")