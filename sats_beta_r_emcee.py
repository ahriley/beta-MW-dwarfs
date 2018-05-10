import numpy as np
import yaml
import emcee
import corner

study = 'fritz'

ndim = 4    # mu_alpha, mu_delta, v_LOS, dist
def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

# read in dwarfs yaml
dwarf_file = 'data/dwarfs/'+study+'.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

# for each dwarf, run MCMC chain for 4-D space
positions = []
for name in names:
    d = dwarfs[name]
    means = np.array([d['mu_alpha'],d['mu_delta'],d['vel_los'],d['Distance']])
    cov = [[d['mu_alpha_error']**2,d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error'],0,0],
            [d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error'],d['mu_delta_error']**2,0,0],
            [0,0,d['vel_los_error']**2,0],
            [0,0,0,d['Distance_error']**2]]
    cov = np.array(cov)
    icov = np.linalg.inv(cov)

    nwalkers = 250
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
    pos, prob, state = sampler.run_mcmc(p0, 300)
    sampler.reset()

    pos, prob, state = sampler.run_mcmc(pos, 1000)
    """
    # plot it
    figure = corner.corner(sampler.flatchain, labels=[r"$\mu_\alpha$",
                            r"$\mu_\delta$", r"$v_{LOS}$", r"$d$"],
                            quantiles=[0.16, 0.5, 0.84], title_fmt='.3f',
                            show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig('figures/mcmc/'+name+'.png', bbox_inches='tight')
    """
    positions.append(pos)
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
