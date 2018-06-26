import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import yaml
import likelihood.fullcov as l

# get indices of unwanted satellites
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

# load MC samples, remove unwanted satellites
MC_dwarfs = np.load('data/sampling/fritz_converted.npy')
MC_dwarfs = MC_dwarfs[:,:,9:12]
MC_dwarfs = np.swapaxes(MC_dwarfs,0,1)

# data and covariances for each satellite
vels = np.mean(MC_dwarfs, axis=1)
vel_covs = np.array([np.cov(np.swapaxes(dwarf,0,1)) for dwarf in MC_dwarfs])

# Initialize walkers by randomly sampling prior
ndim, nwalkers = 9, 100
p0 = l.sample_prior(ndim=ndim, nwalkers=nwalkers)

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob, args=(vels, vel_covs))
pos, prob, state = sampler.run_mcmc(p0, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,1]);

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

# if needed, reset and run chain for new sample
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 500)

# Flatten the chain and remove burn-in
burnin = 0
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:,3:6] = 10**samples[:,3:6]
samples[:,6] *= samples[:,3]*samples[:,4]
samples[:,7] *= samples[:,3]*samples[:,5]
samples[:,8] *= samples[:,4]*samples[:,5]

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$",
                        r"$cov_{r\theta}$", r"$cov_{r\phi}$", r"$cov_{\theta\phi}$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/uniform_fullcov.png', bbox_inches='tight')
np.save('data/mcmc/mcmc_uniform_fullcov', samples, allow_pickle=False)
