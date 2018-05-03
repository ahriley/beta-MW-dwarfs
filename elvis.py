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

all = np.array(all).flatten()
topvmax = np.array(topvmax).flatten()
topvpeak = np.array(topvpeak).flatten()

fulldata = np.concatenate((all, topvmax, topvpeak))
edges = np.linspace(min(fulldata), max(fulldata), 100)

kwargs = {'histtype':'step', 'cumulative':True, 'bins':edges, 'lw':2.0, 'density':True}
plt.hist(all, **kwargs, label='allsubs')
plt.hist(topvmax, **kwargs, label='top 50 vmax')
plt.hist(topvpeak, **kwargs, label='top 50 vpeak')
plt.title("CDF of beta for subhalos in ELVIS")
plt.xlabel('Beta')
plt.ylabel('Fraction of main halos')
plt.legend(loc='upper left');
