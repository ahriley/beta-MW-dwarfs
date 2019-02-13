import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import yaml
import likelihood.variable_simple as l
# import likelihood.uniform_simple as l
import pickle
import utils as u

# load MC samples, names of satellites
sample = 'fritzplusMCs'
tag = 'fritzplusMCs'
scaled_errors = True

scaletag = '_scalederrs' if scaled_errors else ''
sample += scaletag
tag += scaletag
MC_dwarfs = np.load('data/sampling/'+sample+'.npy')
with open('data/sampling/names_key'+scaletag+'.pkl', 'rb') as f:
    names = pickle.load(f)[sample]
assert MC_dwarfs.shape[0] == len(names)

"""
# cut based on distances
dists = MC_dwarfs[:,6,:]
dists = np.median(dists, axis=1)
inc = dists > 100
MC_dwarfs = MC_dwarfs[inc]
# """

"""
# use satellites from Cautun & Frenk (2017)
cautun = ['Sagittarius I', 'LMC', 'SMC', 'Draco I', 'Ursa Minor', 'Sculptor',
            'Carina I', 'Fornax', 'Leo II', 'Leo I']
cautun = np.array([names.index(sat) for sat in cautun])
MC_dwarfs = MC_dwarfs[cautun]
# """

"""
# ignore satellites by name
ignoresats = ['Horologium I', 'Carina II', 'Carina III', 'Hydrus I']
# ignoresats = ['LMC', 'SMC']
# ignoresats = ['Horologium I', 'Carina II', 'Carina III', 'Hydrus I', 'LMC',
#                 'SMC']
# ignoresats = ['Sagittarius I']
ignore = [names.index(sat) for sat in ignoresats]
MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)
# """

"""
# cut based on brightness
absmag = []
with open('data/dwarf_props.yaml', 'r') as f:
    dwarfs = yaml.load(f)
    for name in names:
        absmag.append(dwarfs[name]['abs_mag'])
absmag = np.array(absmag)
inc = absmag < -7.7         # < means brighter, > means fainter
MC_dwarfs = MC_dwarfs[inc]
# """

# data and covariances for each satellite
MC_vels = MC_dwarfs[:,9:12,:]
vels = np.mean(MC_vels, axis=2)
vel_covs = np.array([np.cov(dwarf) for dwarf in MC_vels])
dists = MC_dwarfs[:,6,:]
dists = np.median(dists, axis=1)

# Initialize walkers by randomly sampling prior
nwalkers = 100
p0 = l.sample_prior(nwalkers=nwalkers)
ndim = len(p0[0])

# Set up and run MCMC
# sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob, args=(vels,vel_covs))
sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob, args=(vels,vel_covs,dists))
pos, prob, state = sampler.run_mcmc(p0, 1000)

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
# fig = corner.corner(samples, labels=[r"$v_\phi$", r"$\sigma_r$", \
#                         r"$\sigma_\theta = \sigma_\phi$"],
#                       quantiles=[0.16, 0.5, 0.84],
#                       show_titles=True, title_kwargs={"fontsize": 12})

fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                    r"$\sigma_{0,r}$", r"$\sigma_{0,\theta} = \sigma_{0,\phi}$",
                    r"$r_{0,r}$", r"$r_{0,\theta} = r_{0,\phi}$",
                    r"$\alpha_r$", r"$\alpha_\theta = \alpha_\phi$"],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/cornerplots/'+tag+'.png', bbox_inches='tight')
np.save(u.SIM_DIR+'beta/mcmc/data/'+tag, samples)
