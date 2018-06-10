import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import corner
import utils as u
"""
# compute velocity ellipsoids
# vr vtheta vphi sigma_r sigma_theta sigma_phi cov_rtheta cov_rphi cov_thetaphi
samples = np.load('data/mcmc/mcmc_uniform_fullcov.npy')
sigma = samples[:,3:6]
cov = samples[:,6:]

data = np.zeros((50000,6))
data[:,0] = cov[:,0]/(sigma[:,0]*sigma[:,1])
data[:,1] = cov[:,1]/(sigma[:,0]*sigma[:,2])
data[:,2] = cov[:,2]/(sigma[:,1]*sigma[:,2])
data[:,3] = np.arctan(2*cov[:,0]/(sigma[:,0]**2-sigma[:,1]**2))*90/np.pi
data[:,4] = np.arctan(2*cov[:,1]/(sigma[:,0]**2-sigma[:,2]**2))*90/np.pi
data[:,5] = np.arctan(2*cov[:,2]/(sigma[:,1]**2-sigma[:,2]**2))*90/np.pi

fig = corner.corner(data, labels=[r"corr$_{r\theta}$", r"corr$_{r\phi}$", r"corr$_{\theta\phi}$",
                        r"$\alpha_{r\theta}$", r"$\alpha_{r\phi}$", r"$\alpha_{\theta\phi}$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})
fig.savefig('figures/velocityellipsoid_corner.png', bbox_inches='tight')
# """

"""
# posterior for beta from uniform gaussian model
samples = np.load('data/mcmc/mcmc_uniform_fullcov.npy')
betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
plt.hist(betas, bins=50, density=True, label='fullcov', histtype='step', lw=2)
samples = np.load('data/mcmc/mcmc_uniform_baseline.npy')
betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
plt.hist(betas, bins=50, density=True, label='orig (diagonal)', histtype='step', lw=2)
plt.xlabel(r'$\beta$')
plt.legend(loc='upper left')
plt.title(r"Posterior for $\beta$");
plt.savefig('figures/beta_posterior_diagonalvsfullcov.png', bbox_inches='tight');

col = np.sort(betas)
ixL = np.floor(np.size(col)*.159).astype(int)
ixU = np.floor(np.size(col)*.841).astype(int)
lower_mc = col[ixL]
upper_mc = col[ixU]

med = np.median(col)
print(med, lower_mc-med, upper_mc - med)
# """

"""
# plot of beta posterior with variable sigmas
samples = np.load('data/mcmc/mcmc_variablesigma_constrainedprior.npy')

# variable dispersions with distance
def sigma(r, sigma0, r0, alpha):
    return sigma0*(1+(r/r0))**-alpha

rvals = np.arange(0,300,5)
sigmas = [sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) for r in rvals]
sigmas = np.array(sigmas)
betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) for i in range(len(rvals))]
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

plt.plot(rvals, beta_median, '-', lw=2.0)
plt.fill_between(rvals, lower, upper, alpha = 0.4)
plt.axhline(y=0, ls='--', c='k')
plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$\beta$')
# plt.ylim(-3,1)
plt.title(r"Posterior for $\beta(r)$");
plt.savefig('figures/beta_posterior_variablesigma_constrainedprior.png', bbox_inches='tight');
# """

"""
# plot of beta posterior with variable sigmas for multiple sims

# variable dispersions with distance
def sigma(r, sigma0, r0, alpha):
    return sigma0*(1+(r/r0))**-alpha

list = u.list_of_sims('elvis')
list.append('satellites')

for sim in list:
    if sim == 'satellites':
        samples = np.load('data/mcmc/mcmc_variablesigma_constrainedprior.npy')
    else:
        samples = np.load('data/mcmc/elvis_variablesigma_vpeak/'+sim+'.npy')

    rvals = np.arange(0,300,5)
    sigmas = [sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) for i in range(len(rvals))]
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

    plt.plot(rvals, beta_median, '-', lw=2.0)
    if sim == 'satellites':
        plt.fill_between(rvals, lower, upper, alpha = 0.4)
    else:
        plt.fill_between(rvals, lower, upper, alpha = 0.2)
plt.axhline(y=0, ls='--', c='k')
plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$\beta$')
# plt.ylim(-3,1)
plt.title(r"Posterior for $\beta(r)$, top 40 Vpeak");
plt.savefig('figures/elvis_variablesigma_vpeak/beta_posteriors.png', bbox_inches='tight');
# """

# """
# plot of beta posterior with variable sigmas for multiple sims

# variable dispersions with distance
def sigma(r, sigma0, r0, alpha):
    return sigma0*(1+(r/r0))**-alpha

list = u.list_of_sims('elvis')
cuts = ['sats', '', '_vmax', '_vpeak', '_vmax7']
labels = [None, 'vmax', 'vpeak', 'vmax7']
rvals = np.arange(0,300,5)

# satellite data
samples = np.load('data/mcmc/mcmc_variablesigma_constrainedprior.npy')

sigmas = [sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) for r in rvals]
sigmas = np.array(sigmas)
betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) for i in range(len(rvals))]
betas = np.array(betas)
data = np.median(betas, axis=1)

for sim in list:
    for cut, label in zip(cuts, labels):
        if cut == 'sats':
            samples = np.load('data/mcmc/mcmc_variablesigma_constrainedprior.npy')
        else:
            samples = np.load('data/mcmc/elvis_variablesigma'+cut+'/'+sim+'.npy')
        sigmas = [sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) for r in rvals]
        sigmas = np.array(sigmas)
        betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) for i in range(len(rvals))]
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

        plt.plot(rvals, beta_median-data, '-', lw=2.0, label=label)
        plt.fill_between(rvals, lower-data, upper-data, alpha = 0.2)
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$\beta$')
    plt.legend(loc='upper right')
    plt.title(r"Posterior for $\beta(r)$, "+sim);
    plt.savefig('figures/elvis_variablesigma_normed/'+sim+'.png', bbox_inches='tight');
    plt.close()
# """

"""
# histogram of Galactocentric velocity errors
# get indices of unwanted satellites
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())
ignore = [names.index('Cra I'), names.index('Eri II'), names.index('Phe I')]

# load MC samples, remove unwanted satellites
MC_dwarfs = np.load('data/mcmc/sampling_converted_fritz.npy')
MC_dwarfs = MC_dwarfs[:,:,9:12]
MC_dwarfs = np.swapaxes(MC_dwarfs,0,1)
MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)

# data and covariances for each satellite
vels = np.mean(MC_dwarfs, axis=1)
vel_covs = np.array([np.cov(np.swapaxes(dwarf,0,1)) for dwarf in MC_dwarfs])
errors = np.array([np.sqrt(np.diag(cov)) for cov in vel_covs])

edges = np.arange(0,600,10)
plt.hist(errors[:,0], histtype='step', bins=edges, label=r"$r$", lw=2.0)
plt.hist(errors[:,1], histtype='step', bins=edges, label=r"$\theta$", lw=2.0)
plt.hist(errors[:,2], histtype='step', bins=edges, label=r"$\phi$", lw=2.0)
plt.title(r"Satellites w/ $\delta_i < \delta$")
plt.xlabel(r"$\delta$ [km/s]")
plt.ylabel(r"$N$")
plt.legend(loc='lower right')
# plt.xlim(0,200)
plt.savefig('figures/cdf_galactocentric_errors2.png', bbox_inches='tight');
"""

"""
# specifics of coordinate transformation
import yaml
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

dsph = dwarfs[names[0]]
sc = SkyCoord(ra=dsph['RA']*u.degree, dec=dsph['DEC']*u.degree,
                distance=dsph['Distance']*u.kpc,
                pm_ra_cosdec=dsph['mu_alpha']*u.mas/u.yr,
                pm_dec=dsph['mu_delta']*u.mas/u.yr,
                radial_velocity=dsph['vel_los']*u.km/u.s, frame='icrs')
sc = sc.transform_to(coord.Galactocentric)
"""

"""
# me being terrified I have coordinates wrong
import utils as u
sats = u.load_satellites('data/dwarfs/fritz_cart.csv')
assert np.isclose(sats.x**2+sats.y**2+sats.z**2, sats.r**2).all()
assert np.isclose(sats.z, sats.r*np.cos(sats.theta)).all()
assert np.isclose(sats.x, sats.r*np.sin(sats.theta)*np.cos(sats.phi)).all()
assert np.isclose(sats.y, sats.r*np.sin(sats.theta)*np.sin(sats.phi)).all()
"""
