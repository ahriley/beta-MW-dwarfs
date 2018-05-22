import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import yaml

# import samples generated from below
samples = np.loadtxt('data/mcmc/uniform_gaussians.txt')
# np.savetxt('data/mcmc/uniform_gaussians.txt', samples, header='v_r v_theta v_phi sigma_r sigma_theta sigma_phi')

betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
plt.hist(betas, bins=50, density=True)
plt.xlabel(r'$\beta$')
plt.title(r"Posterior for $\beta$")
plt.savefig('figures/beta_posterior_uniformgaussaians.png', bbox_inches='tight');

"""
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
    cov_theta = np.diag(theta[3:]**2)
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

# Compute maximum likelihood params
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, np.random.rand(6)*100, args=(vels, vel_covs))

# Prior (flat)
def lnprior(theta):
    m = theta[:3]
    s = theta[3:]
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
ndim, nwalkers = 6, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up and run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vels, vel_covs))
sampler.run_mcmc(pos, 500)

# Look by eye at the burn-in
stepnum = np.arange(0,500,1)+1
stepnum = np.array([stepnum for i in range(nwalkers)])
plt.plot(stepnum, sampler.chain[:,:,5]);

# Flatten the chain and remove burn-in
burnin = 100
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Make corner plot
fig = corner.corner(samples, labels=[r"$v_r$", r"$v_\theta$", r"$v_\phi$",
                        r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$"],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})
# fig.savefig("triangle.png")
# Extract the axes
axes = np.array(fig.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(result["x"][i], color="g")

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(result["x"][xi], color="g")
        ax.axhline(result["x"][yi], color="g")
        ax.plot(result["x"][xi], result["x"][yi], "sg")

fig.savefig('figures/likelihood.png', bbox_inches='tight')
"""

"""
samples[:, 2] = np.exp(samples[:, 2])
vr_mcmc, vtheta_mcmc, vphi_mcmc, sigr_mcmc, sigtheta_mcmc, sigphi_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
"""
