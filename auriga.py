import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import utils as u
import likelihood.variablesigma as l
import pandas as pd
import glob
from os.path import isfile

auriga_path = u.SIM_DIR+'auriga/'
names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'Mstar', 'Vmax']
sims = glob.glob(auriga_path+'*.txt')
label = 'gt_5e6_rnum'

for sim in sims:
    name = sim.strip(auriga_path).strip('.txt')
    folder = 'DMO' if 'DMO' in name else 'auriga'

    # check this analysis hasn't been done
    outfile = u.SIM_DIR+'beta/mcmc/'+folder+'/'+name+'_'+label
    if isfile(outfile+'.npy'):
        print("File computed for "+sim+", haloID: "+str(ID))
        continue

    subs = pd.read_table(sim, sep='\s+', header=None, names=names)
    subs = u.compute_spherical_hostcentric(subs)
    print(name+": "+str(len(subs))+" subhalos")

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
