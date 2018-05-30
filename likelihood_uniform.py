import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import yaml

# get indices of unwanted satellites
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())
ignore = [names.index('Cra I'), names.index('Eri II'), names.index('Phe I')]
"""
cautun_dwarfs = ['Sgr I', 'Dra I', 'U Min I', 'Scu I', 'Car I', 'Frn I',
                    'Leo I', 'Leo II', 'LMC', 'SMC']
names.append("LMC")
names.append("SMC")
inc = [names.index(dwarf) for dwarf in cautun_dwarfs]
"""
# load MC samples, remove unwanted satellites
MC_dwarfs = np.load('data/mcmc/sampling_converted_fritz.npy')
# MC_clouds = np.load('data/mcmc/sampling_converted_magclouds.npy')
# MC_dwarfs = np.concatenate((MC_dwarfs,MC_clouds), axis=1)
# dists = MC_dwarfs[:,:,6]
MC_dwarfs = MC_dwarfs[:,:,9:12]
MC_dwarfs = np.swapaxes(MC_dwarfs,0,1)
MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)

# data and covariances for each satellite
vels = np.mean(MC_dwarfs, axis=1)
vel_covs = np.array([np.cov(np.swapaxes(dwarf,0,1)) for dwarf in MC_dwarfs])
# dists = np.median(np.delete(np.swapaxes(dists,0,1), ignore, axis=0), axis=1)
# inc = dists < 100
# vels = vels[inc]
# vel_covs = vel_covs[inc]

# Likelihood
def lnlike(theta, data, data_covs):
    shifts = data - theta[:3]
    cov_theta = np.diag((10**theta[3:])**2)
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

# Prior (flat)
def lnprior(theta):
    m = theta[:3]
    lns = theta[3:]
    if (m<500).all() and (m>-500).all() and (lns<3).all() and (lns>-3).all():
        return 0.0
    return -np.inf

# Full log-probability function (likelihood*prior)
def lnprob(theta, data, data_covs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs)

# Initialize walkers by randomly sampling prior
ndim, nwalkers = 6, 100
p_scale = np.array([1000,1000,1000,6,6,6])
p_shift = np.array([500,500,500,3,3,3])
p0 = [np.random.uniform(size=ndim)*p_scale - p_shift for i in range(nwalkers)]

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vels, vel_covs))
pos, prob, state = sampler.run_mcmc(p0, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,0]);

# if needed, reset and run chain for new sample
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 500)

# Flatten the chain and remove burn-in
burnin = 200
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:,3:] = 10**samples[:,3:]

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/uniform_baseline.png', bbox_inches='tight')
np.save('data/mcmc/mcmc_uniform_baseline', samples, allow_pickle=False)
