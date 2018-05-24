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
def lnlike(theta, data, data_covs):
    shifts = data - theta[:3]
    sig_r, sig_theta, sig_phi, cov_rtheta, cov_rphi, cov_thetaphi = theta[3:]
    cov_theta = [[sig_r**2, cov_rtheta, cov_rphi],
                    [cov_rtheta, sig_theta**2, cov_thetaphi],
                    [cov_rphi, cov_thetaphi, sig_phi**2]]
    cov_theta = np.array(cov_theta)
    if np.linalg.det(cov_theta) <= 0:
        return -np.inf
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

# Compute maximum likelihood params
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, np.random.rand(9)*100, args=(vels, vel_covs))

# Prior (flat)
def lnprior(theta):
    m = theta[:3]
    s = theta[3:6]
    if (m<500).all() and (m>-500).all() and (s<500).all() and (s>0).all():
        return 0.0
    return -np.inf

# Full log-probability function (likelihood*prior)
def lnprob(theta, data, data_covs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs)

# Initialize walkers in tiny ball around max-likelihood result
ndim, nwalkers = 9, 100
p0 = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vels, vel_covs))
pos, prob, state = sampler.run_mcmc(p0, 500)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,8]);

# Flatten the chain and remove burn-in
burnin = 0
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$",
                        r"$cov_{r\theta}$", r"$cov_{r\phi}$", r"$cov_{\theta\phi}$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig('figures/uniform_fullcov.png', bbox_inches='tight')
np.save('data/mcmc/mcmc_uniform_fullcov', samples, allow_pickle=False)
