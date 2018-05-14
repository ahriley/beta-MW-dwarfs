import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as u

num = 50
edges = np.array([0,50,100,150,200,300])
rvals = (edges[1:] + edges[:-1])/2

"""
# separating into mass bins
lowers = [10**6, 10**7, 10**8, 10**9]
uppers = [10**7, 10**8, 10**9, 10**10]
labels = [r'$10^6 - 10^7\ M_\odot$', r'$10^7 - 10^8\ M_\odot$',
            r'$10^8 - 10^9\ M_\odot$', r'$10^9 - 10^{10}\ M_\odot$']

for lower, upper, label in zip(lowers, uppers, labels):
"""

beta_profiles = []
for sim in u.list_of_sims(suite='elvis'):
    subs = u.load_elvis(sim=sim)

    # redshift-dependent cut
    # z, subs_z = u.get_halos_at_redshift(sim=sim, z_target=7)
    # subs = subs.loc[subs_z.nlargest(n=num, columns='Vmax').index]

    # identify main halos, separate from rest
    subs.sort_values('M_dm', ascending=False, inplace=True)
    haloIDs = list(subs.index.values[0:2])
    subs, halos = subs.drop(haloIDs), subs.loc[haloIDs]
    halos.sort_values('M_star', ascending=False, inplace=True)
    And_id = halos.iloc[0].name
    MW_id = halos.iloc[1].name

    # grab subhalos of main halos
    subs = subs[(subs['hostID'] == And_id) | (subs['hostID'] == MW_id)]
    # subs = subs[(subs['M_dm'] > lower) & (subs['M_dm'] < upper)]

    # center on main halos, convert to spherical coordinates
    subs = u.center_on_hosts(hosts=halos, subs=subs)
    subs.x, subs.y, subs.z = subs.x*u.Mpc2km, subs.y*u.Mpc2km, subs.z*u.Mpc2km
    subs = u.compute_spherical_hostcentric_sameunits(df=subs)
    subs.x, subs.y, subs.z = subs.x*u.km2kpc, subs.y*u.km2kpc, subs.z*u.km2kpc
    subs.r = subs.r*u.km2kpc

    for ID in haloIDs:
        halo_subs = subs[subs.hostID == ID]
        halo_subs = halo_subs.nlargest(n=num, columns='Vpeak')
        cut = pd.cut(halo_subs['r'], bins=edges, labels=False)
        beta = np.empty(0)
        for i in range(len(rvals)):
            beta = np.append(beta, u.beta(halo_subs[cut == i]))
            # if np.sum(cut == i) < 5:
            #     beta[-1] = np.nan
        beta_profiles.append(beta)
beta_profiles = np.array(beta_profiles)
median = np.nanmedian(beta_profiles, axis=0)

# confidence intervals
lower, upper = [np.empty(0) for i in range(2)]
for i in range(len(beta_profiles[0])):
    col = beta_profiles[:,i]
    col = np.sort(col[~np.isnan(col)])
    ixL = np.floor(np.size(col)*.159).astype(int)
    ixU = np.floor(np.size(col)*.841).astype(int)
    lower = np.append(lower, col[ixL])
    upper = np.append(upper, col[ixU])

# values directly from measurements
sats = u.load_satellites('data/dwarfs/fritz_cart.csv')
sats = sats[(sats.r < 300) & (sats.index != 'Cra I')]
cut = pd.cut(sats['r'], bins=edges, labels=False)
beta_sats = np.empty(0)
for i in range(len(rvals)):
    beta_sats = np.append(beta_sats, u.beta(sats[cut == i]))

mcmc_betas = np.loadtxt('data/mcmc/beta_profile_fritz_biglastbin.txt')
mcmc_medians = np.nanmedian(mcmc_betas, axis=0)
mcmc_means = np.nanmean(mcmc_betas, axis=0)

# confidence intervals
lower_mc, upper_mc = [np.empty(0) for i in range(2)]
for i in range(len(mcmc_betas[0])):
    col = mcmc_betas[:,i]
    col = np.sort(col[~np.isnan(col)])
    ixL = np.floor(np.size(col)*.159).astype(int)
    ixU = np.floor(np.size(col)*.841).astype(int)
    lower_mc = np.append(lower_mc, col[ixL])
    upper_mc = np.append(upper_mc, col[ixU])

# plt.plot(rvals, mcmc_medians, 'o')
# plt.fill_between(rvals, lower, upper, alpha=0.4, color='C0', label='68% CI')
yerr = np.array([mcmc_medians-lower_mc,upper_mc-mcmc_medians])
xerr = np.array([rvals-edges[:-1], edges[1:]-rvals])
plt.errorbar(rvals, mcmc_medians, yerr=yerr, xerr=xerr,fmt='o', label=r'median, $1\sigma$')
plt.plot(rvals, beta_sats, 'o', label='measurements')

# plot the things
plt.plot(rvals, median, lw=3.0)
plt.fill_between(rvals, lower, upper, alpha = 0.4)
# plt.title(r'ELVIS top 50 Vmax at z='+"{:.2f}".format(z))
plt.title("ELVIS top 50 Vpeak")
plt.xlabel("Galactocentric distance [kpc]")
plt.ylim(-5, 1)
plt.ylabel(r'$\beta$');
plt.savefig("figures/beta_r_elvisvpeak_sats.png", bbox_inches='tight')
