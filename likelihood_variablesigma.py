import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import likelihood.variablesigma as l

sample = 'fritz'
ignore = []

"""
# ignore probable LMC satellites
names_sample = sample.strip('fritz_') if 'fritz_' in sample else sample
# with open('data/sampling/'+names_sample+'_key.pkl', 'rb') as f:
#     import pickle
#     names = pickle.load(f)['name']
# with open('data/dwarfs/'+sample+'.yaml') as f:
#     import yaml
#     names = list(yaml.load(f).keys())
LMC_sats = ['Horologium I', 'Carina II', 'Carina III', 'Hydrus I']
[ignore.append(names.index(sat)) for sat in LMC_sats]
# """

# load MC samples, remove unwanted satellites
MC_dwarfs = np.load('data/sampling/'+sample+'_converted.npy')
dists = MC_dwarfs[:,6,:]
MC_dwarfs = MC_dwarfs[:,9:12,:]

if len(ignore) > 0:
    MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)
    dists = np.delete(dists, ignore, axis=0)

# data and covariances for each satellite
vels = np.mean(MC_dwarfs, axis=2)
vel_covs = np.array([np.cov(dwarf) for dwarf in MC_dwarfs])
dists = np.mean(dists, axis=1)

# Initialize walkers by randomly sampling prior
ndim, nwalkers = 12, 100
p0 = l.sample_prior(ndim=ndim, nwalkers=nwalkers)

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob, args=(vels, vel_covs, dists))
pos, prob, state = sampler.run_mcmc(p0, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,10]);

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

# if needed, reset and run chain for new sample
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 500)

# Flatten the chain and remove burn-in
burnin = 0
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:,3:6] = 10**samples[:,3:6]

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$",
                        r"$r_{0,r}$", r"$r_{0,\theta}$", r"$r_{0,\phi}$",
                        r"$\alpha_r$", r"$\alpha_\theta$", r"$\alpha_\phi$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/cornerplots/variablesigma_fritz_HPMKS_noLMCsats.png', bbox_inches='tight')
np.save('data/mcmc/variablesigma_fritz_HPMKS_noLMCsats', samples)
