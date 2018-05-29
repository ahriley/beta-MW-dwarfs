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

# load MC samples, remove unwanted satellites
MC_dwarfs = np.load('data/mcmc/sampling_converted_fritz.npy')
MC_dwarfs = MC_dwarfs[:,:,9:12]
MC_dwarfs = np.swapaxes(MC_dwarfs,0,1)
MC_dwarfs = np.delete(MC_dwarfs, ignore, axis=0)

# data and covariances for each satellite
vels = np.mean(MC_dwarfs, axis=1)
vel_covs = np.array([np.cov(np.swapaxes(dwarf,0,1)) for dwarf in MC_dwarfs])

# Likelihood
# NOTE: finish up making this based off of correlations
def lnlike(theta, data, data_covs):
    shifts = data - theta[:3]
    sig_r, sig_theta, sig_phi = 10**theta[3:6]
    rtheta, rphi, thetaphi = theta[6:]
    cov_theta = [[sig_r**2, rtheta*sig_r*sig_theta, rphi*sig_r*sig_phi],
                    [rtheta*sig_r*sig_theta, sig_theta**2, thetaphi*sig_theta*sig_phi],
                    [rphi*sig_r*sig_phi, thetaphi*sig_theta*sig_phi, sig_phi**2]]
    cov_theta = np.array(cov_theta)
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    if np.any([np.linalg.det(covs) < 0 for cov in data_covs]):
        return -np.inf
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

# Prior (flat)
def lnprior(theta):
    m = theta[:3]
    lns = theta[3:6]
    corr = theta[6:]
    if ((m<500).all() and (m>-500).all() and (lns<3).all() and
        (lns>-3).all() and (corr<=1).all() and (corr>=-1).all()):
        return 0.0
    return -np.inf

# Full log-probability function (likelihood*prior)
def lnprob(theta, data, data_covs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs)

# Initialize walkers by randomly sampling prior
ndim, nwalkers = 9, 100
p_scale = np.array([1000,1000,1000,6,6,6,2,2,2])
p_shift = np.array([500,500,500,3,3,3,1,1,1])
p0 = [np.random.uniform(size=ndim)*p_scale - p_shift for i in range(nwalkers)]
p0 = np.array(p0)

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vels, vel_covs))
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
