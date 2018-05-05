import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as u

num = 50
all, topvmax, topvpeak = [], [], []
# sim = 'Hall&Oates'
for sim in u.list_of_sims(suite='elvis'):
    subs = u.load_elvis(sim=sim)

    # identify main halos, separate from rest
    subs.sort_values('M_dm', ascending=False, inplace=True)
    haloIDs = list(subs.index.values[0:2])
    subs, halos = subs.drop(haloIDs), subs.loc[haloIDs]
    halos.sort_values('M_star', ascending=False, inplace=True)
    And_id = halos.iloc[0].name
    MW_id = halos.iloc[1].name

    # grab subhalos of main halos
    subs = subs[(subs['hostID'] == And_id) | (subs['hostID'] == MW_id)]

    # center on main halos, convert to spherical coordinates
    subs = u.center_on_hosts(hosts=halos, subs=subs)
    subs.x, subs.y, subs.z = subs.x*u.Mpc2km, subs.y*u.Mpc2km, subs.z*u.Mpc2km
    subs = u.compute_spherical_hostcentric_sameunits(df=subs)
    subs.x, subs.y, subs.z = subs.x*u.km2kpc, subs.y*u.km2kpc, subs.z*u.km2kpc
    subs.r = subs.r*u.km2kpc

    Andsubs = subs[subs.hostID == And_id]
    MWsubs = subs[subs.hostID == MW_id]

    all.append([u.beta(Andsubs), u.beta(MWsubs)])
    topvmax.append([u.beta(Andsubs.nlargest(n=num, columns='Vmax')), u.beta(MWsubs.nlargest(n=num, columns='Vmax'))])
    topvpeak.append([u.beta(Andsubs.nlargest(n=num, columns='Vpeak')), u.beta(MWsubs.nlargest(n=num, columns='Vpeak'))])

all = np.array(all)
topvmax = np.array(topvmax)
topvpeak = np.array(topvpeak)

fulldata = np.concatenate((all, topvmax, topvpeak)).flatten()
edges = np.linspace(min(fulldata), max(fulldata), 100)

kwargs = {'histtype':'step', 'cumulative':True, 'bins':edges, 'lw':2.0, 'density':True}
plt.figure(figsize=(8,6))
plt.hist(all.flatten(), **kwargs, label='all subs of main halo')
plt.hist(topvmax.flatten(), **kwargs, label='top 50 vmax')
plt.hist(topvpeak.flatten(), **kwargs, label='top 50 vpeak')
plt.title("CDF of beta for subhalos in ELVIS")
plt.xlabel('Beta')
plt.ylabel('Fraction of main halos')
plt.legend(loc='upper left');
#plt.savefig('figures/cdf_beta_vmaxandvpeakcuts.png', bbox_inches='tight')
#plt.close()

plt.figure(figsize=(8,6))
plt.hist(all[:,0], ls='-', color='C0', **kwargs)
plt.hist(topvmax[:,0], ls='-', color='C1', **kwargs)
plt.hist(topvpeak[:,0], ls='-', color='C2', **kwargs)
plt.hist(all[:,1], ls=':', color='C0', **kwargs)
plt.hist(topvmax[:,1], ls=':', color='C1', **kwargs)
plt.hist(topvpeak[:,1], ls=':', color='C2', **kwargs)
plt.title("CDF of beta for subhalos in ELVIS")
plt.xlabel('Beta')
plt.ylabel('Fraction of main halos');
#plt.savefig('figures/cdf_beta_vmaxandvpeakcuts_sep.png', bbox_inches='tight')

plt.figure(figsize=(8,6))
plt.plot(topvmax[:,1], topvpeak[:,1], 'o', label='MW')
plt.plot(topvmax[:,0], topvpeak[:,0], 'o', label='And')
plt.plot([-0.8, 0.4], [-0.8, 0.4], '-')
plt.legend(loc='best')
plt.xlabel('Beta (top 50 Vmax)')
plt.ylabel('Beta (top 50 Vpeak)')
plt.title('Overall beta for Vmax vs. Vpeak cuts');
#plt.savefig('figures/overallbeta_vpeak_vmax.png', bbox_inches='tight')
