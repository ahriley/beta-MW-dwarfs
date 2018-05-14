import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import utils as u

samples = np.load('data/mcmc/sampling_converted_fritz.npy')
outfile = 'data/mcmc/beta_profile_fritz_biglastbin.txt'
edges = np.array([0,50,100,150,200,300])
rvals = (edges[1:] + edges[:-1]) / 2
head = str(edges)

# get dwarf names, column names
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())
colnames = ['vx', 'vy', 'vz', 'x', 'y', 'z', 'r', 'theta', 'phi', 'v_r',
            'v_theta', 'v_phi', 'v_t']

beta_profiles = []
for sample in samples:
    sats = pd.DataFrame(data=sample, index=names, columns=colnames)
    sats = sats[(sats.index!='Cra I')&(sats.index!='Eri II')&(sats.index!='Phe I')]
    sats = sats[sats['r'] < 300]
    sats['bin'] = pd.cut(sats['r'], bins=edges, labels=False)

    beta_profile, r_median = [], []
    for i in np.sort(sats['bin'].unique()):
        subset = sats[sats['bin'] == i]
        beta_profile.append(u.beta(subset))
    beta_profiles.append(np.array(beta_profile))

beta_profiles = np.array(beta_profiles)
np.savetxt(fname=outfile, X=beta_profiles, header=head)

# mcmc_betas = np.loadtxt('data/mcmc/beta_profile_fritz.txt')
mcmc_medians = np.nanmedian(beta_profiles, axis=0)
mcmc_means = np.nanmean(beta_profiles, axis=0)

# confidence intervals
lower, upper = [np.empty(0) for i in range(2)]
for i in range(len(beta_profiles[0])):
    col = beta_profiles[:,i]
    col = np.sort(col[~np.isnan(col)])
    ixL = np.floor(np.size(col)*.159).astype(int)
    ixU = np.floor(np.size(col)*.841).astype(int)
    lower = np.append(lower, col[ixL])
    upper = np.append(upper, col[ixU])

"""
# get values directly from measurements
sats = u.load_satellites('data/dwarfs/fritz_cart.csv')
sats = sats[(sats.r < 300) & (sats.index != 'Cra I')]
cut = pd.cut(sats['r'], bins=edges, labels=False)
beta_sats = np.empty(0)
for i in range(len(rvals)):
    beta_sats = np.append(beta_sats, u.beta(sats[cut == i]))
"""
rvals = (edges[1:] + edges[:-1]) / 2
# plt.plot(rvals, mcmc_medians, 'o')
# plt.fill_between(rvals, lower, upper, alpha=0.4, color='C0', label='68% CI')
yerr = np.array([mcmc_medians-lower,upper-mcmc_medians])
plt.errorbar(rvals, mcmc_medians, yerr=yerr, fmt='o', label=r'median, $1\sigma$')
# plt.plot(rvals, beta_sats, 'o', label='measurements')
plt.legend(loc='best')
plt.ylim(-5,1)
plt.title("Beta profile from Fritz ")
plt.xlabel("Galactocentric distance [kpc]")
plt.ylabel(r'$\beta$');
# plt.savefig("figures/beta_r_fritz_mc.png", bbox_inches='tight')

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
