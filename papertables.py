import numpy as np
import utils as u
import pandas as pd
import pickle

# # ## ### ##### ######## ############# #####################
### Simulation properties
# # ## ### ##### ######## ############# #####################

colnames = ['sim', 'haloID', 'Mvir', 'Rvir', 'Mstar', 'Rstar', 'Vcirc']
haloprops = pd.read_csv('data/APOSTLE_mainHalos.txt', sep='\s+', \
                    names=colnames)
haloprops.set_index(['sim', 'haloID'], inplace=True)
haloprops['Mvir'] /= 10**12
haloprops['Mstar'] /= 10**10

simlist = ['V1_HR_fix', 'V4_HR_fix', 'V6_HR_fix', 'S4_HR_fix', 'S5_HR_fix']
namelist = ['AP-01', 'AP-04', 'AP-06', 'AP-10', 'AP-11']
for sim, name in zip(simlist, namelist):
    halos, subs_c = u.load_apostle(sim=sim, processed=True)

    # treat each APOSTLE halo separately
    first = True
    for ID in halos.index:
        subs = subs_c[subs_c.hostID == ID]
        halo = haloprops.loc[sim,ID]
        propstring = "{:0.2f}".format(halo.Mvir) + ' & ' + \
                        "{:0.1f}".format(halo.Rvir) + ' & ' + \
                        "{:0.1f}".format(halo.Mstar) + ' & ' + \
                        "{:0.1f}".format(halo.Rstar) + ' & ' + \
                        "{:0.1f}".format(halo.Vcirc) + ' & ' + \
                        str(np.sum(subs.Vmax > 5)) + ' & ' + \
                        str(np.sum(subs.Mstar > 0))
        start = '\multirow{2}{*}{'+name+'} & ' if first else '& '
        first = False
        print(start+propstring+' \\\\')

colnames = ['sim', 'Mvir', 'Rvir', 'Mstar', 'Rstar', 'z_scale', 'Vcirc']
haloprops = pd.read_csv('data/AURIGA_mainHalos.txt', sep='\s+', \
                    names=colnames)
haloprops['Mvir'] /= 10**12
haloprops['Mstar'] /= 10**10

simlist = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27']
namelist = ['Au6', 'Au16', 'Au21', 'Au23', 'Au24', 'Au27']
for sim, name in zip(simlist, namelist):
    subs = u.load_auriga(sim=sim, processed=True)
    try:
        halo = haloprops.loc[sim]
    except KeyError:
        halo = haloprops.loc[sim.replace('_', '')]
    propstring = "{:0.2f}".format(halo.Mvir) + ' & ' + \
                    "{:0.1f}".format(halo.Rvir) + ' & ' + \
                    "{:0.1f}".format(halo.Mstar) + ' & ' + \
                    "{:0.1f}".format(halo.Rstar) + ' & ' + \
                    "{:0.1f}".format(halo.Vcirc) + ' & ' + \
                    str(np.sum(subs.Vmax > 5)) + ' & ' + \
                    str(np.sum(subs.Mstar > 0))
    start = name+' & '
    print(start+propstring+' \\\\')

# # ## ### ##### ######## ############# #####################
### Velocity parameters
# # ## ### ##### ######## ############# #####################
sample = 'fritzplusMCs'
MC_dwarfs = np.load('data/sampling/'+sample+'.npy')
MC_pars = MC_dwarfs[:,6:12,:]
with open('data/sampling/names_key.pkl', 'rb') as f:
    names = pickle.load(f)[sample]

MC_pars[:,1,:] *= 180/np.pi
MC_pars[:,2,:] *= 180/np.pi

med = np.percentile(MC_pars, 50, axis=2)
lower = np.percentile(MC_pars, 16, axis=2)
upper = np.percentile(MC_pars, 84, axis=2)

for name in sorted(names):
    sat = names.index(name)
    print(name+' & ', end='')
    for i in range(med.shape[1]):
        if i == med.shape[1]-1:
            print("${:0.1f}".format(med[sat][i])+\
                    '_{-'+"{:0.1f}".format(med[sat][i]-lower[sat][i])+'}'\
                    '^{+'+"{:0.1f}".format(upper[sat][i]-med[sat][i])+'}$'+\
                    ' \\\\')
        else:
            print("${:0.1f}".format(med[sat][i])+\
                    '_{-'+"{:0.1f}".format(med[sat][i]-lower[sat][i])+'}'\
                    '^{+'+"{:0.1f}".format(upper[sat][i]-med[sat][i])+'}$ & ',\
                    end='')
