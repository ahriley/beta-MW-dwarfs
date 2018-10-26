import numpy as np
import matplotlib.pyplot as plt
import utils as u
import yaml
import glob
import pickle
import corner
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import rc
from halotools.empirical_models import NFWProfile

pltpth = 'figures/paperfigs/'
params = {'axes.labelsize': 8,
            'font.size': 8,
            'legend.fontsize': 7,
            'xtick.labelsize':8,
            'ytick.labelsize':8,
            'text.usetex': True,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'legend.numpoints': 1,
            'xtick.top': True,
            'xtick.direction': 'in',
            'ytick.right': True,
            'ytick.direction': 'in'}
plt.rcParams.update(params)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage{amsmath}'+'\n'+r'\usepackage{amssymb}')

pt_to_in = 0.01384
smallwidth = 242.26653 * pt_to_in
largewidth = 513.11743 * pt_to_in

# remake = [True for i in range(8)]
remake = [False for i in range(9)]
remake[7] = True

# # ## ### ##### ######## ############# #####################
### Tangential velocity excess
# # ## ### ##### ######## ############# #####################
if remake[0]:
    print("Tangential velocity excess")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])

    # get name, luminosity of satellites
    with open('data/sampling/names_key.pkl', 'rb') as f:
        names = pickle.load(f)['fritzplusMCs']

    Lstar = []
    with open('data/dwarf_props.yaml', 'r') as f:
        dwarfs = yaml.load(f)
        for name in names:
            Lstar.append(10**(-0.4*(dwarfs[name]['abs_mag'] - 4.83)))
    Lstar = np.array(Lstar)

    # load sampling (Fritz + Magellanic Clouds)
    MC_dwarfs = np.load('data/sampling/fritzplusMCs.npy')
    dists = MC_dwarfs[:,6,:]
    v_r = MC_dwarfs[:,9,:]
    v_t = MC_dwarfs[:,12,:]
    frac = v_r**2/(v_r**2 + v_t**2)

    frac_med = np.median(frac, axis=1)
    dist_med = np.median(dists, axis=1)
    dist_err = [dist_med - np.percentile(dists, 15.9, axis=1),
                np.percentile(dists, 84.1, axis=1) - dist_med]
    frac_err = [frac_med - np.percentile(frac, 15.9, axis=1),
                np.percentile(frac, 84.1, axis=1) - frac_med]

    plt.scatter(dist_med, frac_med, c=np.log10(Lstar), cmap='plasma_r', s=20)
    plt.colorbar().set_label(r'$\log_{10}(L_\ast/{\rm L}_\odot)$')
    plt.errorbar(dist_med, frac_med, fmt='none', yerr=frac_err, xerr=dist_err, \
                    zorder=0, lw=1)
    plt.axhline(1/3, color='k', ls='--', lw=1.0)
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$V_\mathrm{rad}^2\ /\ V_\mathrm{tot}^2$')
    plt.savefig(pltpth+'vtan_excess.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Circular velocity profiles
# # ## ### ##### ######## ############# #####################
if remake[1]:
    print("Circular velocity profiles")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])

    # APOSTLE
    files = glob.glob('data/sims/Vcirc_APOSTLE/*.txt')
    legend = False
    for file in files:
        profile = np.loadtxt(file)
        if not legend:
            plt.plot(profile[:,0], profile[:,1], 'C0', label='APOSTLE')
            legend = True
        else:
            plt.plot(profile[:,0], profile[:,1], 'C0')
        # plt.plot(profile[:,0], profile[:,2], 'r--')

    # Auriga
    files = glob.glob('data/sims/Vcirc_auriga/*.txt')
    legend = False
    for file in files:
        profile = np.loadtxt(file)
        plt.plot(profile[:,0], profile[:,1], 'C1')
        if not legend:
            plt.plot(profile[:,0], profile[:,1], 'C1', label='Auriga')
            legend = True
        else:
            plt.plot(profile[:,0], profile[:,1], 'C1')
        # plt.plot(profile[:,0], profile[:,2], 'b--')

    nfw = NFWProfile()
    nfw_Vcirc = nfw.circular_velocity(profile[:,0]*10**-3, 10**12, conc=10)
    plt.plot(profile[:,0], nfw_Vcirc, 'k--', \
                label=r'NFW(10$^{12}$ M$_\odot$, c=10)')

    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$V_{circ}$ [km s$^{-1}$]')
    plt.xlim(0.5, 150)
    plt.ylim(50, 300)
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(pltpth+'vcirc.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Radial distribution
# # ## ### ##### ######## ############# #####################
if remake[2]:
    print("Radial distribution")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])
    bins = np.logspace(np.log10(5), np.log10(300), 10000)
    rvals = 10**((np.log10(bins[1:])+np.log10(bins[:-1]))/2)

    # Milky Way satellites
    MC_dwarfs = np.load('data/sampling/fritzplusMCs.npy')
    dists = np.median(MC_dwarfs[:,6,:], axis=1)
    MW_dist = np.cumsum(np.histogram(dists, bins=bins)[0])/len(dists)

    # APOSTLE subhalos
    simlist = ['V1_HR_fix', 'V4_HR_fix', 'V6_HR_fix', 'S4_HR_fix', 'S5_HR_fix']
    baryons, corrected = [], []
    for sim in simlist:
        halos, subs_c = u.load_apostle(sim=sim, processed=True)
        subs_c = subs_c[subs_c.Vmax > 5]

        # treat each APOSTLE halo separately
        for ID in halos.index:
            subs = subs_c[subs_c.hostID == ID]
            bary_cdf = np.cumsum(np.histogram(subs.r,bins=bins)[0])/len(subs.r)
            baryons.append(bary_cdf)
            subs = u.match_rdist(subs, 'fritzplusMCs', rtol=10)
            corr_cdf = np.cumsum(np.histogram(subs.r,bins=bins)[0])/len(subs.r)
            corrected.append(corr_cdf)
            # plt.plot(rvals, bary_cdf, 'C0-', lw=1.0)
            # plt.plot(rvals, corr_cdf, 'C0--', lw=1.0)

    # Auriga subhalos
    sims = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27']
    baryons_a, corrected_a = [], []
    for sim in sims:
        subs = u.load_auriga(sim, processed=True)
        subs = subs[subs.Vmax > 5]
        bary_cdf = np.cumsum(np.histogram(subs.r, bins=bins)[0])/len(subs.r)
        baryons_a.append(bary_cdf)
        subs = u.match_rdist(subs, 'fritzplusMCs', rtol=10)
        corr_cdf = np.cumsum(np.histogram(subs.r, bins=bins)[0])/len(subs.r)
        corrected.append(corr_cdf)
        # plt.plot(rvals, bary_cdf, 'C1-', lw=1.0)
        # plt.plot(rvals, corr_cdf, 'C1--', lw=1.0)

    baryons = np.array(baryons); corrected = np.array(corrected)
    baryons_a = np.array(baryons_a); corrected_a = np.array(corrected_a)

    baryon_median = np.percentile(baryons, 50, axis=0)
    baryon_upper = np.percentile(baryons, 100, axis=0)
    baryon_lower = np.percentile(baryons, 0, axis=0)
    corrected_median = np.percentile(corrected, 50, axis=0)
    corrected_upper = np.percentile(corrected, 100, axis=0)
    corrected_lower = np.percentile(corrected, 0, axis=0)

    baryon_median_a = np.percentile(baryons_a, 50, axis=0)
    baryon_upper_a = np.percentile(baryons_a, 100, axis=0)
    baryon_lower_a = np.percentile(baryons_a, 0, axis=0)

    plt.plot(rvals, MW_dist, color='k', label='Milky Way', zorder=100)
    plt.plot(rvals, baryon_median, label=r'APOSTLE', color='C0')
    plt.fill_between(rvals, baryon_lower, baryon_upper, alpha=0.5)
    plt.plot(rvals, baryon_median_a, label=r'Auriga', color='C1')
    plt.fill_between(rvals, baryon_lower_a, baryon_upper_a, color='C1',\
                        alpha=0.5)
    plt.plot(rvals, corrected_median, color='magenta', label='Corrected')
    plt.fill_between(rvals, corrected_lower, corrected_upper, color='violet',\
                        alpha=0.5)
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$f(<r)$')
    plt.xlim(5, 300)
    plt.ylim(10**-4, 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower right');
    plt.savefig(pltpth+'radial_dist.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Uniform results for MW
# # ## ### ##### ######## ############# #####################
if remake[3]:
    print("Uniform results for MW")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])
    bins = np.linspace(-3, 1, 50)
    kwargs = {'bins': bins, 'density': True, 'histtype': 'step', 'lw': 2}
    files = ['uniform_simple', 'uniform_simple_lt100', 'uniform_simple_gt100']
    labels = ['all', r'$r<100$ kpc', r'$r>100$ kpc']
    # files = ['uniform']
    # labels = [None]

    # compute statistics for Cautun sample
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/uniform_simple_cautun.npy')
    betas = 1 - (samples[:,2]**2 + samples[:,2]**2) / (2*samples[:,1]**2)
    lower = np.percentile(betas, 15.9)
    upper = np.percentile(betas, 84.1)
    median = np.median(betas)

    plt.axvspan(lower, upper, ymax=0.27, color='b', alpha=0.2)
    plt.axvline(median, ymax=0.266, color='b')
    plt.axvspan(-2.6, -1.8, ymax=0.27, color='0.6', alpha=0.5)
    plt.axvline(-2.2, ymax=0.266, color='k')
    for file, label in zip(files, labels):
        samples = np.load(u.SIM_DIR+'beta/mcmc/data/'+file+'.npy')
        betas = 1 - (samples[:,2]**2 + samples[:,2]**2) / (2*samples[:,1]**2)
        y, x = np.histogram(betas, bins=bins)
        plt.plot((x[1:] + x[:-1]) / 2,y/len(betas), label=label)
    plt.axvline(0.0, color='k', ls='--')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Posterior distribution')
    plt.xlim(-3,1)
    plt.ylim(bottom=0.0)
    plt.legend(loc='upper left')
    plt.text(-2.26,0.03,'Cautun \& Frenk\n(2017)',ha='center',fontsize=7);
    plt.savefig(pltpth+'uniform.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Corner plot for uniform model
# # ## ### ##### ######## ############# #####################
if remake[4]:
    print("Uniform model corner plot")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/uniform_simple.npy')
    labels = [r"$v_\phi$", r"$\sigma_r$", r"$\sigma_\theta = \sigma_\phi$"]
    params['xtick.labelsize'] = 14
    params['ytick.labelsize'] = 14
    plt.rcParams.update(params)
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, plot_datapoints=False,
                          title_fmt='.0f', title_kwargs={"fontsize": 20},
                          label_kwargs={"fontsize": 20})
    fig.savefig(pltpth+'corner_uniform.pdf', bbox_inches='tight')
    plt.close()
    params['xtick.labelsize'] = 8
    params['ytick.labelsize'] = 8
    plt.rcParams.update(params)

# # ## ### ##### ######## ############# #####################
### Variable results for data
# # ## ### ##### ######## ############# #####################
if remake[5]:
    print('Variable results for MW')
    sample = 'variable_simple'
    fig = plt.figure(figsize=[smallwidth, smallwidth])
    rvals = np.arange(15,265,5)
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/'+sample+'.npy')
    sigmas = [u.sigma(r, samples[:,1:3], samples[:,3:5], samples[:,5:]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,1]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median = np.median(betas, axis=1)

    lower = np.percentile(betas, 15.9, axis=1)
    upper = np.percentile(betas, 84.1, axis=1)

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=0.0)

    ax0 = plt.subplot(gs[0])
    ax0.plot(rvals, beta_median, '-')
    ax0.fill_between(rvals, lower, upper, alpha=0.2)
    ax0.axhline(y=0, ls='--', c='k')
    ax0.set_xlim(rvals[0], rvals[-1])
    ax0.set_ylim(top=1)
    ax0.set_ylabel(r'$\beta$')
    ax0.set_xscale('log')

    ax1 = plt.subplot(gs[1])
    labels = [r'$\sigma_r$', r'$\sigma_\theta=\sigma_\phi$']
    for j in range(2):
        betas = sigmas[:,:,j]
        beta_median = np.median(betas, axis=1)
        lower = np.percentile(betas, 15.9, axis=1)
        upper = np.percentile(betas, 84.1, axis=1)

        ax1.plot(rvals, beta_median, '-', label=labels[j])
        ax1.fill_between(rvals, lower, upper, alpha = 0.4)

    # plot satellites on graph
    MC_dwarfs = np.load('data/sampling/fritzplusMCs.npy')
    dists = np.median(MC_dwarfs[:,6,:], axis=1)
    for dist in dists:
        # ax0.axvline(dist, ls='--', lw=0.5, dashes=(10, 10))
        # ax1.axvline(dist, ls='--', lw=0.5, dashes=(10, 10))
        ax0.axvline(dist, lw=0.5, ymin=0, ymax=0.1)
        ax1.axvline(dist, lw=0.5, ymin=0.9, ymax=1)

    ax1.set_xlabel(r'$r$ [kpc]')
    ax1.set_ylabel(r'$\sigma$ [km s$^{-1}$]')
    ax1.legend(loc=(0.67,0.6))
    ax1.set_xscale('log')
    ax1.set_yticks([0, 50, 100, 150, 200, 250])
    ax1.set_xlim(rvals[0], rvals[-1])
    plt.savefig(pltpth+'variable.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Beta(r) in simulations
# # ## ### ##### ######## ############# #####################
if remake[6]:
    print("Simulation results")
    cols = ['DMO', 'apostle', 'auriga']
    colnames = ['DMO', 'APOSTLE', 'Auriga']
    rows = ['Vmax', 'Vmax_rdist', 'Vmax_rnum', 'Mstar']
    rownames = [r'All',
                r'Radial dist.',
                r'Radial dist. and $N_\text{sats}$',
                r'$M_\text{star} > 0$']
    # rows = ['Vmax', 'Vmax_rnum']
    # rownames = [r'All', r'Radial dist. and $N_\text{sats}$']
    rvals = np.arange(15,265,5)

    # only need to calculate sats curves once
    file = u.SIM_DIR+'beta/mcmc/data/variable_simple.npy'
    samples = np.load(file)
    sigmas = [u.sigma(r, samples[:,1:3], samples[:,3:5], samples[:,5:]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,1]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median_sats = np.median(betas, axis=1)
    lower_sats = np.percentile(betas, 15.9, axis=1)
    upper_sats = np.percentile(betas, 84.1, axis=1)

    fig, ax = plt.subplots(len(rows), len(cols), sharex='col', sharey='row', \
                            figsize=(12,10))
                            # figsize=[largewidth, largewidth])
    plt.subplots_adjust(wspace=0.1, hspace=0.13)
    text_dict = {'ha': 'center', 'va': 'center', 'fontsize': 20}
    fig.text(0.5, 0.07, r'$r$ [kpc]', **text_dict)
    fig.text(0.05, 0.5, r'$\beta$', rotation='vertical', **text_dict);
    for i, row, rowname in zip(range(len(rows)), rows, rownames):
        for j, col, colname in zip(range(len(cols)), cols, colnames):
            cax = ax[i,j]
            cax.set_ylim(-3, 1)
            cax.axhline(0.0, color='k', ls='--')
            cax.tick_params(axis='both', which='major', labelsize=14)
            cax.set_xscale('log')
            if j == 0:
                cax.set_ylabel(rowname, fontsize=12)
            if i == 0:
                cax.set_title(colname, fontsize=12)

            cax.plot(rvals, beta_median_sats, 'k--', zorder=100, lw=2)
            cax.fill_between(rvals, lower_sats, upper_sats, \
                                alpha=0.2, zorder=100, color='k')

            # plot curves for each simulation selection
            simlist = glob.glob(u.SIM_DIR+'beta/mcmc/'+col+'/*'+row+'.npy')
            for sim in simlist:
                samples = np.load(sim)
                sigmas = [u.sigma(r, samples[:,3:6], samples[:,6:9], \
                            samples[:,9:12]) for r in rvals]
                sigmas = np.array(sigmas)
                betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/ \
                            (2*sigmas[i,:,0]**2) for i in range(len(rvals))]
                betas = np.array(betas)
                beta_median = np.median(betas, axis=1)
                lower = np.percentile(betas, 15.9, axis=1)
                upper = np.percentile(betas, 84.1, axis=1)

                cax.plot(rvals, beta_median, '-', zorder=0)
                cax.fill_between(rvals, lower, upper, alpha=0.2, zorder=0)
    plt.savefig(pltpth+'beta_sims.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Different tracer comparison
# # ## ### ##### ######## ############# #####################
if remake[7]:
    print("Different tracers")
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])
    ms = 4
    capsize = 2

    # plot beta(r) for the MW satelites
    sample = 'variable_simple'
    rvals = np.arange(15,265,5)
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/'+sample+'.npy')
    sigmas = [u.sigma(r, samples[:,1:3], samples[:,3:5], samples[:,5:]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,1]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median = np.median(betas, axis=1)
    lower = np.percentile(betas, 15.9, axis=1)
    upper = np.percentile(betas, 84.1, axis=1)

    plt.plot(rvals, beta_median, '-', c='C0', label='This work')
    plt.fill_between(rvals, lower, upper, alpha=0.2, color='C0')

    # Sohn et al. 2018 (HST GCs)
    rval = 10**((np.log10(39.5) + np.log10(10.6))/2)
    plt.errorbar([rval], [.609], yerr=np.array([[0.229], [0.13]]), \
                    xerr=np.array([[rval-10.6], [39.5-rval]]), c='dimgrey', fmt='^',\
                    ms=ms, label='Sohn+18', capsize=capsize, zorder=100);

    # Watkins et al. 2018 (Gaia CGs)
    rval = 10**((np.log10(21.1) + np.log10(2.0))/2)
    plt.errorbar([rval], [.48], yerr=np.array([[0.2], [0.15]]), \
                    xerr=np.array([[rval-2.0], [21.1-rval]]), c='dimgrey', fmt='s',\
                    ms=ms, label='Watkins+18', capsize=capsize, zorder=100)

    # load GC data from Vasiliev
    gc_file = u.SIM_DIR+'beta/othertracers/fig7.txt'
    col_names = ['r', 'r16', 'r50', 'r84', 't16', 't50', 't84', 'p16', 'p50',
                    'p84', 'vphi16', 'vphi50', 'vphi84']
    gcs = pd.read_csv(gc_file, sep='\s+')
    gcs.drop(columns=['upp.3'], inplace=True)
    gcs.columns = col_names

    # sample assuming Gaussian at each radius (good enough)
    betas = []
    for ii, row in gcs.iterrows():
        mean = [row.r50, row.t50, row.p50]
        sigma = [np.mean([row.r50-row.r16, row.r84-row.r50])**2,
                    np.mean([row.t50-row.t16, row.t84-row.t50])**2,
                    np.mean([row.p50-row.p16, row.p84-row.p50])**2]
        cov = np.diag(sigma)
        sample = np.random.multivariate_normal(mean=mean, cov=cov, size=100000)

        beta = 1 - (sample[:,1]**2 + sample[:,2]**2) / (2*sample[:,0]**2)
        betas.append(beta)
    betas = np.array(betas)

    # plot beta(r) for GCs
    beta_mid = np.percentile(betas, 50, axis=1)
    beta_low = np.percentile(betas, 16, axis=1)
    beta_high = np.percentile(betas, 84, axis=1)
    plt.plot(gcs.r, beta_mid, c='0.7', label='Vasiliev18')
    plt.fill_between(gcs.r, beta_low, beta_high, alpha=0.2, color='0.5')

    # Cunningham et al. (in prep) -- HALO7D
    rvals = [15, 23, 39]
    betavals = [0.54, 0.64, 0.70]
    errs = [0.15, 0.15, 0.15]
    plt.errorbar(rvals, betavals, yerr=errs, fmt='o', ms=ms, capsize=capsize,\
                    c='k', label='Cunningham+18');

    plt.axhline(y=0, ls='--', c='k')
    plt.ylim(-3, 1)
    plt.xlim(0.5, 265)
    plt.xscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles2 = handles[:2]
    [handles2.append(h[0]) for h in handles[2:]]
    plt.legend(handles2, labels, loc='best')
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$\beta$')
    plt.savefig(pltpth+'tracer_comparison.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Variable results for data
# # ## ### ##### ######## ############# #####################
if remake[8]:
    print('Complex variable results for MW')
    sample = 'fritzplusMCs'
    fig = plt.figure(figsize=[smallwidth, smallwidth])
    rvals = np.arange(15,265,5)
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/'+sample+'.npy')
    sigmas = [u.sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median = np.median(betas, axis=1)

    lower = np.percentile(betas, 15.9, axis=1)
    upper = np.percentile(betas, 84.1, axis=1)

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=0.0)

    ax0 = plt.subplot(gs[0])
    ax0.plot(rvals, beta_median, '-')
    ax0.fill_between(rvals, lower, upper, alpha=0.2)
    ax0.axhline(y=0, ls='--', c='k')
    ax0.set_xlim(rvals[0], rvals[-1])
    ax0.set_ylim(top=1)
    ax0.set_ylabel(r'$\beta$')
    ax0.set_xscale('log')

    ax1 = plt.subplot(gs[1])
    labels = [r'$\sigma_r$', r'$\sigma_\theta$', r'$\sigma_\phi$']
    for j in range(3):
        betas = sigmas[:,:,j]
        beta_median = np.median(betas, axis=1)
        lower = np.percentile(betas, 15.9, axis=1)
        upper = np.percentile(betas, 84.1, axis=1)

        ax1.plot(rvals, beta_median, '-', label=labels[j])
        ax1.fill_between(rvals, lower, upper, alpha = 0.4)

    # plot satellites on graph
    MC_dwarfs = np.load('data/sampling/fritzplusMCs.npy')
    dists = np.median(MC_dwarfs[:,6,:], axis=1)
    for dist in dists:
        # ax0.axvline(dist, ls='--', lw=0.5, dashes=(10, 10))
        # ax1.axvline(dist, ls='--', lw=0.5, dashes=(10, 10))
        ax0.axvline(dist, lw=0.5, ymin=0, ymax=0.1)
        ax1.axvline(dist, lw=0.5, ymin=0.9, ymax=1)

    ax1.set_xlabel(r'$r$ [kpc]')
    ax1.set_ylabel(r'$\sigma$ [km s$^{-1}$]')
    ax1.legend(loc=(0.75,0.5))
    ax1.set_xscale('log')
    ax1.set_yticks([0, 100, 200, 300])
    ax1.set_xlim(rvals[0], rvals[-1])
    plt.savefig(pltpth+'variable_complex.pdf', bbox_inches='tight')
    plt.close()
