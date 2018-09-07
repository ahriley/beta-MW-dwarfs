import numpy as np
import emcee
import utils as u
import likelihood.variablesigma as l
from os.path import isfile

sims = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27',
        'halo_6_DMO', 'halo_16_DMO', 'halo_21_DMO', 'halo_23_DMO',
        'halo_24_DMO', 'halo_27_DMO']
label = 'Mstar'

for sim in sims:
    if label == 'Mstar' and 'DMO' in sim:
        continue

    # check this analysis hasn't been done
    folder = 'DMO' if 'DMO' in sim else 'auriga'
    outfile = u.SIM_DIR+'beta/mcmc/'+folder+'/'+sim+'_'+label
    if isfile(outfile+'.npy'):
        print("File computed for "+sim)
        continue

    subs = u.load_auriga(sim, processed=True)

    # subs = subs[subs.Vmax > 5]
    # subs = subs[subs.Vpeak > 18]
    subs = subs[subs.Mstar > 0]
    # subs = u.match_rdist(subs, 'fritzplusMCs', rtol=10)
    # subs = u.match_rnum(subs, 'fritzplusMCs')
    print(sim+": "+str(len(subs))+" subhalos")

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
    if not 0.2 < accfrac < 0.5:
        print("Mean acceptance fraction for "+sim+": {0:.3f}".format(accfrac))

    # Flatten the chain and remove burn-in
    burnin = 0
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    np.save(outfile, samples)
