import numpy as np
import utils as u

def lnlike(theta, data, data_covs, data_dists):
    shifts = data - np.array([0,0,theta[0]])
    s0s = np.array([theta[1], theta[2], theta[2]])
    r0s = np.array([theta[3], theta[4], theta[4]])
    alphas = np.array([theta[5], theta[6], theta[6]])
    sigmas = np.array([u.sigma(r,s0s,r0s,alphas) for r in data_dists])
    cov_theta = np.array([np.diag(sig**2) for sig in sigmas])
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

def lnprior(theta):
    m = theta[0]
    s0, r0, a = theta[1:3], theta[3:5], theta[5:]
    if ((m<500).all() and (m>-500).all() and (s0>50).all() and (s0<1000).all()
        and (r0>10).all() and (r0<1000).all() and (a>0).all()
        and (a<10).all()):
        return 0.0
    return -np.inf

def lnprob(theta, data, data_covs, data_dists):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs, data_dists)

def sample_prior(nwalkers):
    scale = np.array([1000,950,950,990,990,10,10])
    shift = np.array([500,-50,-50,-10,-10,0,0])
    ndim = len(scale)
    p0 = [np.random.uniform(size=ndim)*scale - shift for i in range(nwalkers)]
    return p0
