import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import utils as u

samples = np.load('data/mcmc/sampling_converted_fritz.npy')
n_per_bin = 9

# get dwarf names, column names
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())
colnames = ['vx', 'vy', 'vz', 'x', 'y', 'z', 'r', 'theta', 'phi', 'v_r',
            'v_theta', 'v_phi', 'v_t']

# assign each dwarf to radial bin (9x4=36)
sats_p = u.load_satellites('data/dwarfs/fritz_cart.csv')
sats_p = sats_p[(sats_p.index!='Cra I')&(sats_p.index!='Eri II')&(sats_p.index!='Phe I')]
sats_p = sats_p.sort_values('r')
sats_p['bin'] = np.floor(np.arange(0,len(sats_p),1)/n_per_bin).astype(int)
"""
# compute beta profile for each sample
beta_profiles, r_medians = [], []
for sample in samples:
    sats = pd.DataFrame(data=sample, index=names, columns=colnames)
    sats = sats[(sats.index!='Cra I')&(sats.index!='Eri II')&(sats.index!='Phe I')]
    sats['bin'] = sats_p['bin']
    assert (np.sort(sats_p['bin'].unique())==np.sort(sats['bin'].unique())).all()

    beta_profile, r_median = [], []
    for i in np.sort(sats['bin'].unique()):
        subset = sats[sats['bin'] == i]
        beta_profile.append(u.beta(subset))
        r_median.append(np.median(subset['r']))
    beta_profiles.append(np.array(beta_profile))
    r_medians.append(np.array(r_median))

beta_profiles = np.array(beta_profiles)
r_medians = np.array(r_medians)
# np.save('data/mcmc/beta_profile_fritz_equalnum', beta_profiles, allow_pickle=False)
# np.save('data/mcmc/beta_profile_fritz_equalnum_rmedian', r_medians, allow_pickle=False)
"""
beta_profiles = np.load('data/mcmc/beta_profile_fritz_equalnum.npy')
r_medians = np.load('data/mcmc/beta_profile_fritz_equalnum_rmedian.npy')

# get median rs in each bin for maximum value from data
beta_p, r_p = [], []
for i in np.sort(sats_p['bin'].unique()):
    subset = sats_p[sats_p['bin'] == i]
    beta_p.append(u.beta(subset))
    r_p.append(np.median(subset['r']))
beta_p = np.array(beta_p)
r_p = np.array(r_p)

# compute medians, CIs for MC sample
beta_medians = np.median(beta_profiles, axis=0)
lower, upper = [np.empty(0) for i in range(2)]
for i in range(len(beta_profiles[0])):
    col = beta_profiles[:,i]
    col = np.sort(col[~np.isnan(col)])
    ixL = np.floor(np.size(col)*.159).astype(int)
    ixU = np.floor(np.size(col)*.841).astype(int)
    lower = np.append(lower, col[ixL])
    upper = np.append(upper, col[ixU])

yerr = np.array([beta_medians-lower,upper-beta_medians])
plt.errorbar(r_p, beta_medians, yerr=yerr, fmt='o', label=r'MC median, $1\sigma$')
plt.plot(r_p, beta_p, 'o', label='peak from data')
plt.legend(loc='best')
plt.title("Beta profile from Fritz (9/bin)")
plt.xlabel("Median Galactocentric distance [kpc]")
plt.ylabel(r'$\beta$');
# plt.savefig('figures/beta_r_9perbin.png', bbox_inches='tight')
