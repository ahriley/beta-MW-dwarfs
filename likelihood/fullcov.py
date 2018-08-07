import numpy as np

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

def lnprior(theta):
    m = theta[:3]
    lns = theta[3:6]
    corr = theta[6:]
    if ((m<500).all() and (m>-500).all() and (lns<3).all() and
        (lns>-3).all() and (corr<=1).all() and (corr>=-1).all()):
        return 0.0
    return -np.inf

def lnprob(theta, data, data_covs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, data_covs)

def sample_prior(nwalkers):
    scale = np.array([1000,1000,1000,6,6,6,2,2,2])
    shift = np.array([500,500,500,3,3,3,1,1,1])
    ndim = len(scale)
    p0 = [np.random.uniform(size=ndim)*scale - shift for i in range(nwalkers)]
    return p0
