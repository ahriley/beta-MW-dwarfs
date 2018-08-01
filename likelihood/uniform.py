import numpy as np

def lnlike(theta, data, data_covs):
    shifts = data - theta[:3]
    cov_theta = np.diag(theta[3:]**2)
    covs = data_covs + cov_theta
    icovs = np.linalg.inv(covs)
    lnlike = np.sum([shift@(icov@shift) for shift,icov in zip(shifts,icovs)])
    lnlike += np.sum(np.log(np.linalg.det(covs)))
    return -lnlike

def lnprior(theta):
    m = theta[:3]
    s = theta[3:]
    if (m<500).all() and (m>-500).all() and (s<300).all() and (s>0).all():
        return 0.0
    return -np.inf

def lnprob(theta, data, data_covs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs)

def sample_prior(ndim, nwalkers):
    scale = np.array([1000,1000,1000,300,300,300])
    shift = np.array([500,500,500,0,0,0])
    p0 = [np.random.uniform(size=ndim)*scale - shift for i in range(nwalkers)]
    return p0
