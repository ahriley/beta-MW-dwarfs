import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
import utils as u
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# sims for which Vpeak is calculated
simlist = ['V1_HR_fix', 'V4_HR_fix', 'V6_HR_fix', 'S5_HR_fix']

for sim in simlist:
    print(sim)
    if 'HR' in sim:
        # APOSTLE
        halos = u.load_apostle(sim)
        if 'Vpeak' in halos.keys():
            print("Already computed for "+sim)
        files = glob.glob(u.APOSTLE_DIR+sim+'/*.sav')
    else:
        # Auriga
        raise NotImplementedError
    tracks_full = np.array([np.loadtxt(file) for file in files])
    Vmax = tracks_full[:,:,2]

    # basic checks
    assert (halos.Vmax > 0).all()
    if sim != 'S5_HR_fix':
        assert (np.abs(halos.Vmax - Vmax[:,-1]) < 0.05).all()
    # breaks for S5_HR_fix b/c I don't have tracks for those main hosts

    # compute Vpeak in three different ways
    Vpeak, Vpeak_raw, Vpeak_spline, Vpeak_savgol = [], [], [], []
    for i in range(len(Vmax)):
        Vmax_i = Vmax[i][Vmax[i] > 0]
        snaps = np.arange(len(Vmax_i))

        # just the biggest value of Vmax
        Vpeak_raw_i = np.max(Vmax_i)
        Vpeak_raw.append(Vpeak_raw_i)

        # interpolate with univariate spline, does a little smoothing
        # note: for noisy signal (e.g. V1_HR_fix i=1) this gets really messy
        if len(Vmax_i) > 3:
            spl = UnivariateSpline(x=snaps, y=Vmax_i)
            x = np.linspace(snaps[0], snaps[-1], 1000)
            Vpeak_spline_i = np.max(spl(x))
        else:
            Vpeak_spline_i = -np.inf
        Vpeak_spline.append(Vpeak_spline_i)

        # signal smoothing that also maintains overall shape of the function
        if len(Vmax_i) > 18:
            savgol_smoothed = savgol_filter(Vmax_i, 19, 3)
            Vpeak_savgol_i = np.max(savgol_smoothed)
        else:
            Vpeak_savgol_i = -np.inf
        Vpeak_savgol.append(Vpeak_savgol_i)

        # based on below plots and  background, savgol smooth preferred
        # note that the difference is minimal (~15 subhalos included/excluded)
        if np.isfinite(Vpeak_savgol_i):
            Vpeak.append(Vpeak_savgol_i)
        else:
            Vpeak.append(Vpeak_raw_i)

    Vpeak_raw = np.array(Vpeak_raw)
    Vpeak_spline = np.array(Vpeak_spline)
    Vpeak_savgol = np.array(Vpeak_savgol)
    Vpeak = np.array(Vpeak)
    assert np.isfinite(Vpeak).all()

    # np.sum((Vpeak_raw > 18) & (Vpeak_savgol < 18))
    # plt.plot(Vpeak_spline, Vpeak_savgol, '.')
    # plt.plot([1,200], [1,200])
    # plt.xscale('log')
    # plt.yscale('log');

    # don't have tracks for main halos in S5_HR_fix
    if sim == 'S5_HR_fix':
        Vpeak = np.insert(Vpeak, 0, np.inf)
        Vpeak = np.insert(Vpeak, 4114, np.inf)
    halos['Vpeak'] = Vpeak
    halos.to_pickle(u.APOSTLE_DIR+sim+'_subs.pkl')
