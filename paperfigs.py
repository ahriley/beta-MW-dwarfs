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

# remake = [True for i in range(7)]
remake = [False for i in range(7)]
remake[1] = True

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
    plt.colorbar().set_label(r'$\log(L_\ast)$ [$L_\odot$]')
    plt.errorbar(dist_med, frac_med, fmt='none', yerr=frac_err, xerr=dist_err, \
                    zorder=0, lw=1)
    plt.axhline(1/3, color='k', ls='--', lw=1.0)
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$V_\mathrm{rad}^2\ /\ V_\mathrm{tot}^2$')
    plt.savefig(pltpth+'vtan_excess.pdf', bbox_inches='tight')
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

    # Auriga subhalos
    sims = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27']
    for sim in sims:
        subs = u.load_auriga(sim, processed=True)
        subs = subs[subs.Vmax > 5]
        bary_cdf = np.cumsum(np.histogram(subs.r, bins=bins)[0])/len(subs.r)
        baryons.append(bary_cdf)
        subs = u.match_rdist(subs, 'fritzplusMCs', rtol=10)
        corr_cdf = np.cumsum(np.histogram(subs.r, bins=bins)[0])/len(subs.r)
        corrected.append(corr_cdf)
    baryons = np.array(baryons); corrected = np.array(corrected)

    baryon_median = np.percentile(baryons, 50, axis=0)
    baryon_upper = np.percentile(baryons, 100, axis=0)
    baryon_lower = np.percentile(baryons, 0, axis=0)
    corrected_median = np.percentile(corrected, 50, axis=0)
    corrected_upper = np.percentile(corrected, 100, axis=0)
    corrected_lower = np.percentile(corrected, 0, axis=0)

    plt.plot(rvals, MW_dist, color='k', label='Milky Way', zorder=100)
    plt.plot(rvals, baryon_median, label=r'Hydro')
    plt.fill_between(rvals, baryon_lower, baryon_upper, alpha=0.5)
    plt.plot(rvals, corrected_median, label='Corrected')
    plt.fill_between(rvals, corrected_lower, corrected_upper, alpha=0.5)
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
### Uniform results for data
# # ## ### ##### ######## ############# #####################
if remake[3]:
    fig = plt.figure()
    print("Uniform sigma")
    bins = np.linspace(-3, 1, 50)
    kwargs = {'bins': bins, 'density': True, 'histtype': 'step', 'lw': 2}
    files = ['uniform', 'uniform_lt100kpc', 'uniform_gt100kpc']
    labels = ['all', '< 100', '> 100']

    # compute statistics for Cautun sample
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/uniform_cautun.npy')
    betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
    lower = np.percentile(betas, 15.9)
    upper = np.percentile(betas, 84.1)
    median = np.median(betas)

    plt.axvspan(-2.6, -1.8, ymax=0.3, color='0.6', alpha=0.5)
    plt.axvline(-2.2, ymax=0.296, color='k')
    plt.axvspan(lower, upper, ymax=0.3, color='b', alpha=0.2)
    plt.axvline(median, ymax=0.296, color='b')
    for file, label in zip(files, labels):
        samples = np.load(u.SIM_DIR+'beta/mcmc/data/'+file+'.npy')
        betas = 1 - (samples[:,4]**2 + samples[:,5]**2) / (2*samples[:,3]**2)
        y, x = np.histogram(betas, bins=bins)
        plt.plot((x[1:] + x[:-1]) / 2,y/len(betas), label=label, lw=2)
    plt.axvline(0.0, color='k', ls='--')
    plt.xlabel(r'$\beta$', fontsize=16)
    plt.ylabel('Posterior distribution', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0.0)
    plt.legend(loc='upper left')
    plt.text(-2.2,0.03,'Cautun & Frenk\n(2017)',ha='center');
    plt.savefig(pltpth+'uniform.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Corner plot for uniform model
# # ## ### ##### ######## ############# #####################
if remake[4]:
    print("Uniform model corner plot")
    samples = np.load(u.SIM_DIR+'beta/mcmc/data/uniform.npy')
    labels = [r"$v_r$", r"$v_\theta$", r"$v_\phi$", r"$\sigma_r$",
                r"$\sigma_\theta$", r"$\sigma_\phi$"]

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_kwargs={"fontsize": 14},
                          label_kwargs={"fontsize": 18})
    fig.savefig(pltpth+'corner_uniform.pdf', bbox_inches='tight')
    fig.close()

# # ## ### ##### ######## ############# #####################
### Variable results for data
# # ## ### ##### ######## ############# #####################
if remake[5]:
    sample = 'fritzplusMCs'
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

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=0.0)

    ax0 = plt.subplot(gs[0])
    ax0.plot(rvals, beta_median, '-', lw=2.0)
    ax0.fill_between(rvals, lower, upper, alpha=0.2)
    ax0.axhline(y=0, ls='--', c='k')
    ax0.tick_params(labelsize=12)
    ax0.set_xlim(rvals[0], rvals[-1])
    ax0.set_ylim(ymax=1)
    ax0.set_ylabel(r'$\beta$', fontsize=16)
    ax0.set_xscale('log')

    ax1 = plt.subplot(gs[1])
    labels = [r'$\sigma_r$', r'$\sigma_\theta$', r'$\sigma_\phi$']
    for j in range(3):
        betas = sigmas[:,:,j]
        beta_median = np.median(betas, axis=1)
        lower = np.percentile(betas, 15.9, axis=1)
        upper = np.percentile(betas, 84.1, axis=1)

        ax1.plot(rvals, beta_median, '-', lw=2.0, label=labels[j])
        ax1.fill_between(rvals, lower, upper, alpha = 0.4)
    ax1.set_xlabel(r'$r$ [kpc]', fontsize=16)
    ax1.set_ylabel(r'$\sigma$ [km s$^{-1}$]', fontsize=16)
    ax1.legend(loc='best', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yticks([50, 100, 150, 200, 250, 300])
    ax1.tick_params(labelsize=12)
    ax1.set_xlim(rvals[0], rvals[-1]);
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
                r'Radial dist and $N_\mathregular{sats}$',
                r'$M_\mathregular{star} > 0$']
    rvals = np.arange(15,265,5)

    # only need to calculate sats curves once
    file = u.SIM_DIR+'beta/mcmc/data/fritzplusMCs.npy'
    samples = np.load(file)
    sigmas = [u.sigma(r, samples[:,3:6], samples[:,6:9], samples[:,9:12]) \
                for r in rvals]
    sigmas = np.array(sigmas)
    betas = [1-(sigmas[i,:,1]**2 + sigmas[i,:,2]**2)/(2*sigmas[i,:,0]**2) \
                for i in range(len(rvals))]
    betas = np.array(betas)
    beta_median_sats = np.median(betas, axis=1)
    lower_sats = np.percentile(betas, 15.9, axis=1)
    upper_sats = np.percentile(betas, 84.1, axis=1)

    fig, ax = plt.subplots(len(rows), len(cols), sharex='col', sharey='row', \
                            figsize=(12,10))
    plt.subplots_adjust(wspace=0.1, hspace=0.13)
    text_dict = {'ha': 'center', 'va': 'center', 'fontsize': 20}
    fig.text(0.5, 0.07, 'r [kpc]', **text_dict)
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

            cax.plot(rvals, beta_median_sats, '-', lw=2.0)
            cax.fill_between(rvals, lower_sats, upper_sats, alpha = 0.2)

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

                cax.plot(rvals, beta_median, '-', lw=2.0)
                cax.fill_between(rvals, lower, upper, alpha = 0.2)
    plt.savefig(pltpth+'beta_sims.pdf', bbox_inches='tight')
    plt.close()

# # ## ### ##### ######## ############# #####################
### Different tracer comparison
# # ## ### ##### ######## ############# #####################
if remake[7]:
    fig = plt.figure(figsize=[smallwidth, smallwidth*0.75])
    ms = 4
    capsize = 2

    # plot beta(r) for the MW satelites
    sample = 'fritzplusMCs'
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

    plt.plot(rvals, beta_median, '-', c='C0', label='This work')
    plt.fill_between(rvals, lower, upper, alpha=0.2, color='C0')

    # Sohn et al. 2018 (HST GCs)
    rval = 10**((np.log10(39.5) + np.log10(10.6))/2)
    plt.errorbar([rval], [.609], yerr=np.array([[0.229], [0.13]]), \
                    xerr=np.array([[rval-10.6], [39.5-rval]]), c='0.5', fmt='^',\
                    ms=ms, label='Sohn+18', capsize=capsize);

    # Watkins et al. 2018 (Gaia CGs)
    rval = 10**((np.log10(21.1) + np.log10(2.0))/2)
    plt.errorbar([rval], [.48], yerr=np.array([[0.2], [0.15]]), \
                    xerr=np.array([[rval-2.0], [21.1-rval]]), c='0.5', fmt='s',\
                    ms=ms, label='Watkins+18', capsize=capsize)

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
    plt.plot(gcs.r, beta_mid, c='0.5', label='Vasiliev18')
    plt.fill_between(gcs.r, beta_low, beta_high, alpha=0.2, color='0.5')

    # Cunningham et al. (in prep) -- HALO7D
    rvals = [15, 23, 35, 39]
    betavals = [0.58, 0.60, 0.59, 0.49]
    errs = [0.15, 0.1, 0.05, 0.15]
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

# plot satellites on graph
# ylim = plt.gca().get_ylim()
# names = [x for _,x in sorted(zip(dist_med,names))]
# dist_med = np.sort(dist_med)
# highs = ['Tuc III', 'Car III', 'Tri II', 'Boo II', 'Com Ber I', 'Boo I',
#             'U Min', 'Scu', 'U Maj I', 'Car I', 'Gru I', 'For', 'Can Ven II']
# for name, dist in zip(names, dist_med):
#     split = name.split()
#     id = [item[:3] for item in split]
#     if 'Ursa' in name:
#         id[0] = id[0][0]
#     tag = ' '.join(id)
#     if tag == 'Sex':
#         tag = 'Sxt'
#     if tag in highs:
#         plt.text(dist, ylim[1] - 0.1, tag, rotation='vertical', va='top')
#     else:
#         plt.text(dist, ylim[0] + 0.1, tag, rotation='vertical', va='bottom')
