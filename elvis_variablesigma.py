import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
import utils as u

# variable dispersions with distance
def sigma(r, sigma0, r0, alpha):
    return sigma0*(1+(r/r0))**-alpha

# Likelihood
def lnlike(theta, data, data_covs, data_dists):
    shifts = data - theta[:3]
    s0s, r0s, alphas = 10**theta[3:6], theta[6:9], theta[9:12]
    sigmas = np.array([sigma(r,s0s,r0s,alphas) for r in data_dists])
    cov_theta = np.array([np.diag(sig**2) for sig in sigmas])
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

def lnlike2(theta, data, data_dists):
    shifts = data - theta[:3]
    s0s, r0s, alphas = 10**theta[3:6], theta[6:9], theta[9:12]
    sigmas = np.array([sigma(r,s0s,r0s,alphas) for r in data_dists])
    lnlike = np.sum(shifts**2/sigmas**2)
    lnlike += np.sum(np.prod(sigmas**2, axis=1))
    return -lnlike

# Prior (flat)
def lnprior(theta):
    m = theta[:3]
    lns0, r0, a = theta[3:6], theta[6:9], theta[9:12]
    if ((m<500).all() and (m>-500).all() and (lns0>-3).all() and (lns0<3).all()
        and (r0>10).all() and (r0<1000).all() and (a>0).all() and (a<10).all()):
        return 0.0
    return -np.inf

# Full log-probability function (likelihood*prior)
def lnprob(theta, data, data_covs, data_dists):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs, data_dists)

def lnprob2(theta, data, data_dists):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta, data, data_dists)

for sim in u.list_of_sims('elvis'):
    print("Computing for "+sim)
    subs = u.load_elvis(sim=sim)
    subs.sort_values('M_dm', ascending=False, inplace=True)
    haloIDs = list(subs.index.values[0:2])
    subs, halos = subs.drop(haloIDs), subs.loc[haloIDs]
    halos.sort_values('M_star', ascending=False, inplace=True)
    And_id = halos.iloc[0].name
    MW_id = halos.iloc[1].name

    # grab subhalos of main halos
    subs = subs[(subs['hostID'] == And_id) | (subs['hostID'] == MW_id)]

    # center on main halos, convert to spherical coordinates
    subs = u.center_on_hosts(hosts=halos, subs=subs)
    subs.x, subs.y, subs.z = subs.x*u.Mpc2km, subs.y*u.Mpc2km, subs.z*u.Mpc2km
    subs = u.compute_spherical_hostcentric_sameunits(df=subs)
    subs.x, subs.y, subs.z = subs.x*u.km2kpc, subs.y*u.km2kpc, subs.z*u.km2kpc
    subs.r = subs.r*u.km2kpc

    vels = subs[['v_r', 'v_theta', 'v_phi']].values
    vel_covs = np.zeros((len(vels),3,3))
    dists = subs.r.values

    # Initialize walkers by randomly sampling prior
    ndim, nwalkers = 12, 100
    p_scale = np.array([1000,1000,1000,6,6,6,990,990,290,10,10,10])
    p_shift = np.array([500,500,500,3,3,3,-10,-10,-10,0,0,0])
    p0 = [np.random.uniform(size=ndim)*p_scale - p_shift for i in range(nwalkers)]

    # Set up and run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vels, vel_covs, dists))
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(vels, dists))
    pos, prob, state = sampler.run_mcmc(p0, 500)

    # if needed, reset and run chain for new sample
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, 500)

    accfrac = np.mean(sampler.acceptance_fraction)
    if not 0.2 < accfrac < 0.5:
        print("Mean acceptance fraction for "+sim+": {0:.3f}".format(accfrac))

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

    fig.savefig('figures/elvis_variablesigma/'+sim+'.png', bbox_inches='tight')
    np.save('data/mcmc/elvis_variablesigma/'+sim, samples, allow_pickle=False)
