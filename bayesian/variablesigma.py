import numpy as np
import utils as u

def lnlike(theta, data, data_covs, data_dists):
    shifts = data - theta[:3]
    s0s, r0s, alphas = 10**theta[3:6], theta[6:9], theta[9:12]
    sigmas = np.array([u.sigma(r,s0s,r0s,alphas) for r in data_dists])
    cov_theta = np.array([np.diag(sig**2) for sig in sigmas])
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

def lnprior(theta):
    m = theta[:3]
    lns0, r0, a = theta[3:6], theta[6:9], theta[9:12]
    if ((m<500).all() and (m>-500).all() and (lns0>-3).all() and (lns0<3).all()
        and (r0>10).all() and (r0<1000).all() and (a>0).all()
        and (a<10).all()):
        return 0.0
    return -np.inf

def lnprob(theta, data, data_covs, data_dists):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs, data_dists)

def sample_prior(ndim, nwalkers):
    scale = np.array([1000,1000,1000,6,6,6,990,990,290,10,10,10])
    shift = np.array([500,500,500,3,3,3,-10,-10,-10,0,0,0])
    p0 = [np.random.uniform(size=ndim)*scale - shift for i in range(nwalkers)]
    return p0
