import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import yaml
import likelihood.uniform as l

# load MC samples
sample = 'fritz'
MC_dwarfs = np.load('data/sampling/'+sample+'_converted.npy')
with open('data/dwarfs/'+sample+'.yaml') as f:
    names = list(yaml.load(f).keys())
if sample == 'fritz':
    # add Magellanic Clouds
    MCs = np.load('data/sampling/helmi_converted.npy')[-2:]
    MC_dwarfs = np.concatenate((MC_dwarfs, MCs))
    names.append('LMC')
    names.append('SMC')

"""
# cut based on distances
dists = MC_dwarfs[:,6,:]
dists = np.median(dists, axis=1)
inc = dists > 100
MC_dwarfs = MC_dwarfs[inc]
# """

# """
# use satellites from Cautun & Frenk (2017)
cautun = ['Sagittarius I', 'LMC', 'SMC', 'Draco I', 'Ursa Minor', 'Sculptor',
            'Carina I', 'Fornax', 'Leo II', 'Leo I']
cautun = np.array([names.index(sat) for sat in cautun])
MC_dwarfs = MC_dwarfs[cautun]
# """

# data and covariances for each satellite
MC_vels = MC_dwarfs[:,9:12,:]
vels = np.mean(MC_vels, axis=2)
vel_covs = np.array([np.cov(dwarf) for dwarf in MC_vels])

# Initialize walkers by randomly sampling prior
nwalkers = 100
p0 = l.sample_prior(nwalkers=nwalkers)

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob, args=(vels,vel_covs))
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

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

tag = 'cautun_sample'
fig.savefig('figures/cornerplots/'+tag+'.png', bbox_inches='tight')
np.save('data/mcmc/'+tag, samples)
