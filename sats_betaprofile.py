import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as u

sats = u.load_satellites('data/dwarfs/fritz_cart.csv')
sats = sats[(sats.r < 300) & (sats.index != 'Cra I')]
edges = np.arange(0,300+1,50)
rvals = (edges[1:] + edges[:-1])/2
cut = pd.cut(sats['r'], bins=edges, labels=False)
beta_sats = np.empty(0)
for i in range(len(rvals)):
    beta_sats = np.append(beta_sats, u.beta(sats[cut == i]))

plt.plot(rvals[:3], beta_sats[:3], 'o');
"""
# plot of number of satellites vs. radius
papers = ['gaia', 'simon', 'fritz']
edges = np.arange(0,450+1,50)
for paper in papers:
    filename = 'data/dwarfs/'+paper+'_cart.csv'
    sats = u.load_satellites(filename)
    plt.hist(sats.r, bins=edges, histtype='step', label=paper, lw=2.0)
plt.legend(loc='best')
plt.xlabel("Galactocentric distance (kpc)")
plt.ylabel("Number of satellites");
# plt.savefig("figures/satellites_distance.png", bbox_inches='tight')
"""
