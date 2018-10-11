import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import likelihood.variablesigma as l
import pickle
import utils as u

# load MC samples, get names
sample = 'fritzplusMCs'
tag = 'fritzplusMCs_noMCsystem'
MC_dwarfs = np.load('data/sampling/'+sample+'.npy')
with open('data/sampling/names_key.pkl', 'rb') as f:
    names = pickle.load(f)[sample]
assert MC_dwarfs.shape[0] == len(names)

# """
# ignore satellites by name
# ignoresats = ['Horologium I', 'Carina II', 'Carina III', 'Hydrus I']
# ignoresats = ['LMC', 'SMC']
ignoresats = ['Horologium I', 'Carina II', 'Carina III', 'Hydrus I', 'LMC',
                'SMC']
ignore = [names.index(sat) for sat in ignoresats]
MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)
# """

dists = MC_dwarfs[:,6,:]
MC_vels = MC_dwarfs[:,9:12,:]

# data and covariances for each satellite
vels = np.median(MC_vels, axis=2)
vel_covs = np.array([np.cov(dwarf) for dwarf in MC_vels])
dists = np.mean(dists, axis=1)

# Initialize walkers by randomly sampling prior
nwalkers = 100
p0 = l.sample_prior_lp(nwalkers=nwalkers)
ndim = len(p0[0])

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob_lp, \
                                args=(vels, vel_covs, dists))
pos, prob, state = sampler.run_mcmc(p0, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,0]);

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

# if needed, reset and run chain for new sample
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 500)

# Flatten the chain and remove burn-in
burnin = 0
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
# samples[:,3:6] = 10**samples[:,3:6]

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$",
                        r"$r_{0,r}$", r"$r_{0,\theta}$", r"$r_{0,\phi}$",
                        r"$\alpha_r$", r"$\alpha_\theta$", r"$\alpha_\phi$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/cornerplots/'+tag+'.png', bbox_inches='tight')
np.save(u.SIM_DIR+'beta/mcmc/data/'+tag, samples)
