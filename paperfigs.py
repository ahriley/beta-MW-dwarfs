import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import utils as u
import yaml

pltpth = 'figures/paperfigs/'

# # ## ### ##### ######## ############# #####################
### Figure 1: Tangential velocity excess
# # ## ### ##### ######## ############# #####################

edges = np.arange(0,301,1)
rhist = {'cumulative': True, 'histtype': 'step', 'lw': 2, 'bins': edges,
            'density': True}

fracbins = np.arange(0,1.01,0.01)
fhist = {'cumulative': True, 'histtype': 'step', 'lw': 2, 'bins': fracbins,
            'density': True}

# get name, luminosity of satellites
with open('data/dwarfs/fritz.yaml', 'r') as f:
    names = list(yaml.load(f).keys())
    names.append('LMC')
    names.append('SMC')

Lstar = []
with open('data/dwarfs/dwarf_props.yaml', 'r') as f:
    dwarfs = yaml.load(f)
    for name in names:
        Lstar.append(10**(-0.4*(dwarfs[name]['abs_mag'] - 4.83)))
Lstar = np.array(Lstar)

# add Magellanic Clouds
MC_dwarfs = np.load('data/sampling/fritz_converted.npy')
MCs = np.load('data/sampling/helmi_converted.npy')[-2:]
MC_dwarfs = np.concatenate((MC_dwarfs, MCs))
dists = MC_dwarfs[:,6,:]
v_r = MC_dwarfs[:,9,:]
v_t = MC_dwarfs[:,12,:]
frac = v_r**2/(v_r**2 + v_t**2)
frac_med = np.median(frac, axis=1)
dist_med = np.median(dists, axis=1)
dist_err = np.std(dists, axis=1)
lower, upper = [np.empty(0) for i in range(2)]
for frac_sample in frac:
    col = np.sort(frac_sample)
    ixL = np.floor(np.size(col)*.159).astype(int)
    ixU = np.floor(np.size(col)*.841).astype(int)
    lower = np.append(lower, col[ixL])
    upper = np.append(upper, col[ixU])
lower = frac_med - lower
upper = upper - frac_med
frac_err = [lower, upper]

plt.scatter(dist_med, frac_med, c=np.log10(Lstar), cmap='plasma_r')
plt.colorbar().set_label(r'$\log(L_\ast)$ [$L_\odot$]')
plt.errorbar(dist_med, frac_med, fmt='none', yerr=frac_err, xerr=dist_err, \
                zorder=0, lw=1.3)
plt.axhline(1/3, color='k', ls='--')
plt.xlabel('Galactocentric dist. [kpc]')
plt.ylabel(r'$V_{r}^2 / V_{tot}^2$');
plt.savefig(pltpth+'vtan_excess.png', bbox_inches='tight');

# # ## ### ##### ######## ############# #####################
### Figure 2: uniform and variable results for data
# # ## ### ##### ######## ############# #####################

plt.figure(figsize=(16,6))
plt.subplot(121)
bins = np.linspace(-3, 1, 50)
kwargs = {'bins': bins, 'density': True, 'histtype': 'step', 'lw': 2}
files = ['fritzplusMCs', 'fritzplusMCs_lt100', 'fritzplusMCs_gt100']
labels = ['all', '< 100', '> 100']
plt.axvspan(-2.6, -1.8, ymax=0.3, color='0.6', alpha=0.5)
plt.axvline(-2.2, ymax=0.296, color='k')
plt.axvspan(-1.38-0.94, -1.23+0.98, ymax=0.3, color='b', alpha=0.2)
plt.axvline(-1.38, ymax=0.296, color='b')
for file, label in zip(files, labels):
    try:
        samples = np.load('data/mcmc/uniform_'+file+'.npy')
    except:
        samples = np.load('data/mcmc/'+file+'.npy')
    betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
    y, x = np.histogram(betas, bins=bins, density=True)
    plt.plot((x[1:] + x[:-1]) / 2,y, label=label, lw=2)
plt.xlabel(r'$\beta$')
plt.ylim(bottom=0.0)
plt.legend(loc='upper left')
plt.text(-2.2,0.4,'Cautun & Frenk\n(2017)',ha='center');

plt.subplot(122)
list = ['fritzplusMCs', 'gold', 'fritz_gold']
colors = ['C0', 'C1', 'C2']
rvals = np.arange(15,265,5)
for sim, color in zip(list, colors):
    samples = np.load('data/mcmc/variablesigma_'+sim+'.npy')
    sigmas = [u.sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median = np.median(betas, axis=1)

    # confidence intervals
    betas_inv = np.swapaxes(betas,0,1)
    lower, upper = [np.empty(0) for i in range(2)]
    for i in range(len(betas_inv[0])):
        col = betas_inv[:,i]
        col = np.sort(col)
        ixL = np.floor(np.size(col)*.159).astype(int)
        ixU = np.floor(np.size(col)*.841).astype(int)
        lower = np.append(lower, col[ixL])
        upper = np.append(upper, col[ixU])

    plt.plot(rvals, beta_median, '-', lw=2.0, label=sim, c=color)
    plt.fill_between(rvals, lower, upper, alpha = 0.2)

    # plot excluding LMC sats as dotted line
    # samples = np.load('data/mcmc/variablesigma_'+sim+'_noLMCsats.npy')
    # sigmas = [u.sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) for r in rvals]
    # sigmas = np.array(sigmas)
    # betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) for i in range(len(rvals))]
    # betas = np.array(betas)
    # beta_median = np.median(betas, axis=1)
    # plt.plot(rvals, beta_median, '--', lw=2.0, c=color)

plt.axhline(y=0, ls='--', c='k')
plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$\beta$')
plt.legend(loc='lower right')
plt.xscale('log');
plt.savefig(pltpth+'uniform_and_variable.png', bbox_inches='tight');