import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import utils as u
import likelihood.variablesigma as l
import pandas as pd
from os.path import isfile

simlist = ['V1_HR_fix', 'V4_HR_fix', 'V6_HR_fix', 'S4_HR_fix', 'S5_HR_fix',
            'V1_HR_DMO', 'V4_HR_DMO']

for sim in simlist:
    halos, subs_c = u.load_apostle(sim=sim, processed=True)
    label = 'Vpeak'

    # if data aren't available, continue
    try:
        # subs_c = subs_c[subs_c.Vmax > 5]
        subs_c = subs_c[subs_c.Vpeak > 18]
    except AttributeError:
        print(sim+" doesn't have the property")
        continue

    # treat each APOSTLE halo separately
    for ID in halos.index:
        # check if this has already been computed
        folder = 'DMO' if 'DMO' in sim else 'apostle'
        outfile = u.SIM_DIR+'beta/mcmc/'+folder+'/'+sim+'_'+str(ID)+'_'+label
        if isfile(outfile+'.npy'):
            print("File computed for "+sim+", haloID: "+str(ID))
            continue

        subs = subs_c[subs_c.hostID == ID]
        # subs = u.match_rdist(subs, 'fritzplusMCs', rtol=10)
        # subs = u.match_rnum(subs, 'fritzplusMCs')
        print(sim+", "+str(ID)+": "+str(len(subs))+" subhalos")

        vels = subs[['v_r', 'v_theta', 'v_phi']].values
        dists = subs.r.values
        vel_covs = np.zeros((len(vels),3,3))

        # Initialize walkers by randomly sampling prior
        nwalkers = 100
        p0 = l.sample_prior_lp(nwalkers=nwalkers)
        ndim = len(p0[0])

        # Set up and run MCMC
        print("Running first 500 steps")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, l.lnprob_lp, \
                                        args=(vels, vel_covs, dists))
        pos, prob, state = sampler.run_mcmc(p0, 500)

        # if needed, reset and run chain for new sample
        sampler.reset()
        print("Running second 500 steps")
        pos, prob, state = sampler.run_mcmc(pos, 500)

        accfrac = np.mean(sampler.acceptance_fraction)
        if not 0.25 < accfrac < 0.5:
            print("Mean acceptance fraction: {0:.3f}".format(accfrac))

        # Flatten the chain and remove burn-in
        burnin = 0
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        # samples[:,3:6] = 10**samples[:,3:6]

        np.save(outfile, samples)
