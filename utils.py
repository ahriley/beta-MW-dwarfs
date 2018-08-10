import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

Mpc2kpc = 1000

ELVIS_DIR = '/Users/alexanderriley/Desktop/sims/elvis/'
APOSTLE_DIR = '/Volumes/TINY/NOTFERMI/sims/apostle/'

def beta(df):
    return 1-(np.var(df.v_theta)+np.var(df.v_phi))/(2*np.var(df.v_r))

def center_on_hosts(hosts, subs):
    centered = subs.copy()
    centered['x'] = subs.x.values - hosts.loc[subs['hostID']].x.values
    centered['y'] = subs.y.values - hosts.loc[subs['hostID']].y.values
    centered['z'] = subs.z.values - hosts.loc[subs['hostID']].z.values
    centered['vx'] = subs.vx.values - hosts.loc[subs['hostID']].vx.values
    centered['vy'] = subs.vy.values - hosts.loc[subs['hostID']].vy.values
    centered['vz'] = subs.vz.values - hosts.loc[subs['hostID']].vz.values

    return centered

def compute_spherical_hostcentric(df):
    x, y, z = df.x, df.y, df.z
    vx, vy, vz = df.vx, df.vy, df.vz

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    v_r = (x*vx + y*vy + z*vz) / r
    v_theta = r*((z*(x*vx+y*vy)-vz*(x**2+y**2)) / (r**2*np.sqrt(x**2+y**2)))
    v_phi = r*np.sin(theta)*((x*vy - y*vx) / (x**2 + y**2))
    v_t = np.sqrt(v_theta**2 + v_phi**2)

    # check that coordinate transformation worked
    cart = np.sqrt(vx**2 + vy**2 + vz**2)
    sphere = np.sqrt(v_r**2 + v_theta**2 + v_phi**2)
    sphere2 = np.sqrt(v_r**2 + v_t**2)
    assert np.allclose(cart, sphere, rtol=1e-03)
    assert np.sum(~np.isclose(cart,sphere))/len(cart) < 0.001
    assert np.allclose(cart, sphere2, rtol=1e-03)
    assert np.sum(~np.isclose(cart,sphere2))/len(cart) < 0.001

    df2 = df.copy()
    df2['r'], df2['theta'], df2['phi'] = r, theta, phi
    df2['v_r'], df2['v_theta'], df2['v_phi'] = v_r, v_theta, v_phi
    df2['v_t'] = v_t

    return df2

def list_of_sims(sim):
    files = []
    if sim == 'elvis':
        files_all = glob.glob(ELVIS_DIR+'*.txt')
        for file in files_all:
            if '&' in file:
                files.append(file)
        return [f[len(ELVIS_DIR):-4] for f in files]
    elif sim == 'apostle':
        files = glob.glob(APOSTLE_DIR+'*.pkl')
        return [f[len(APOSTLE_DIR):-9] for f in files]
    else:
        raise NotImplementedErorr("Specify simulation suite that is available")

def load_apostle(sim, processed=False, sample=None):
    filename = APOSTLE_DIR+sim+'_subs.pkl'
    subs = pd.read_pickle(filename).drop_duplicates()
    if processed:
        subs.sort_values('M_dm', ascending=False, inplace=True)
        haloIDs = list(subs.index.values[0:2])
        subs, halos = subs.drop(haloIDs), subs.loc[haloIDs]
        halos.sort_values('Mstar', ascending=False, inplace=True)
        And_id = halos.iloc[0].name
        MW_id = halos.iloc[1].name
        subs = subs[(subs['hostID'] == And_id) | (subs['hostID'] == MW_id)]
        subs = center_on_hosts(hosts=halos, subs=subs)
        subs.x, subs.y, subs.z = subs.x*Mpc2kpc, subs.y*Mpc2kpc, subs.z*Mpc2kpc
        subs = compute_spherical_hostcentric(df=subs)
        if sample is not None:
            subs = subs[subs[sample]]
        return halos, subs
    return subs

def load_elvis(sim):
    filename = ELVIS_DIR+sim+'.txt'

    # read in the data
    with open(filename) as f:
        id, x, y, z, vx, vy, vz, vmax, vpeak, mvir = [[] for i in range(10)]
        mpeak, rvir, rmax, apeak, mstar, mstar_b = [[] for i in range(6)]
        npart, pid, upid = [], [], []
        for line in f:
            if line[0] == '#':
                continue
            items = line.split()
            id.append(int(items[0]))
            x.append(float(items[1]))
            y.append(float(items[2]))
            z.append(float(items[3]))
            vx.append(float(items[4]))
            vy.append(float(items[5]))
            vz.append(float(items[6]))
            vmax.append(float(items[7]))
            vpeak.append(float(items[8]))
            mvir.append(float(items[9]))
            mpeak.append(float(items[10]))
            rvir.append(float(items[11]))
            rmax.append(float(items[12]))
            apeak.append(float(items[13]))
            mstar.append(float(items[14]))
            mstar_b.append(float(items[15]))
            npart.append(int(items[16]))
            pid.append(int(items[17]))
            upid.append(int(items[18]))

    # convert to pandas format
    df = {'PID': pid, 'hostID': upid, 'npart': npart, 'apeak': apeak,
            'M_dm': mvir, 'Mstar': mstar, 'Mpeak': mpeak, 'Mstar_b': mstar_b,
            'Vmax': vmax, 'Vpeak': vpeak, 'Rvir': rvir, 'Rmax': rmax,
            'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
    df = pd.DataFrame(df, index=id)
    df.index.name = 'ID'
    return df

def get_halos_at_redshift(sim, z_target):
    sim_dir = ELVIS_DIR+'tracks/'+sim+'/'
    a = 1/(1+z_target)

    # find closest scale to target
    with open(sim_dir+'scale.txt') as f:
        scale_list = np.array(f.readlines()[1].split()).astype(float)
    index = np.argmin(np.abs(scale_list - a))
    z_true = 1/scale_list[index] - 1

    # get halo properties at that redshift
    props = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Vmax']
    df = {}
    for prop in props:
        prop_list = []
        with open(sim_dir+prop+'.txt') as f:
            lines = f.readlines()[1:]
            for line in lines:
                split = np.array(line.split()).astype(float)
                prop_list.append(split[index])
        key = prop if prop == 'Vmax' else prop.lower()
        df[key] = prop_list

    # IDs will be for redshift 0
    IDs = []
    with open(sim_dir+'ID.txt') as f:
        lines = f.readlines()[1:]
        for line in lines:
            split = np.array(line.split()).astype(int)
            IDs.append(split[0])

    # TODO: distances changed by scale factor?
    df = pd.DataFrame(df, index=IDs)
    df.index.name = 'ID'
    return z_true, df

def load_satellites(file):
    sats = pd.read_csv(file, index_col=0)
    sats.x, sats.y, sats.z = sats.x*Mpc2km, sats.y*Mpc2km, sats.z*Mpc2km
    sats = compute_spherical_hostcentric_sameunits(df=sats)
    sats.x, sats.y, sats.z = sats.x*km2kpc, sats.y*km2kpc, sats.z*km2kpc
    sats.r = sats.r*km2kpc
    return sats

def match_rdist(df, sample, savefile=None, plotlabel=None,
                plotpath='figures/sampling/match_radial_dists/'):
    colname = sample+'_rdist'
    if colname in df.keys():
        print('Already sampled')
        return df[df[colname]]

    subs_c = df.copy()
    subs_c = subs_c[subs_c.r < 300]

    # get MW satellite distribution
    MC_dwarfs = np.load('data/sampling/'+sample+'_converted.npy')
    dists = np.median(MC_dwarfs[:,6,:], axis=1)

    # find weights for satellites
    edges = np.arange(301, step=50)
    weights = np.histogram(dists, bins=edges)[0] / len(dists)
    labels = np.arange(len(weights))

    # select subhalos
    # NOTE: this is stochastic, so save the results using colname
    subs = subs_c.copy()
    subs['bin'] = np.asarray(pd.cut(subs.r, bins=edges, labels=labels))
    selected = []
    while True:
        try:
            rlbl = np.random.choice(labels, p=weights)
            index = np.random.choice(subs[subs.bin == rlbl].index)
            subs.drop([index], inplace=True)
            selected.append(index)
        except ValueError:
            print(len(selected))
            break
    survived = subs_c.loc[selected]

    # plot if wanted
    if plotlabel is not None:
        plotbins = np.arange(301, step=1)
        k = dict(histtype='step', cumulative=True, bins=plotbins, lw=2)
        w = np.ones_like(survived.r)/float(len(survived.r))
        plt.hist(survived.r, weights=w, label='corrected sim', **k)
        w = np.ones_like(dists)/float(len(dists))
        plt.hist(dists, weights=w, label='MW sats', **k)
        w = np.ones_like(subs_c.r)/float(len(subs_c.r))
        plt.hist(subs_c.r, weights=w, label='original sim', **k)
        plt.legend(loc='upper left')
        plt.savefig(plotpath+plotlabel+'.png', bbox_inches='tight')
        plt.close()

    # save to subhalo data file
    subs_full = pd.read_pickle(savefile).drop_duplicates()
    subs_full[colname] = np.isin(subs_full.index, survived.index)
    assert np.sum(subs_full[colname]) == len(survived)
    assert savefile is not None, "Specify which file result should be saved to"
    subs_full.to_pickle(savefile)
    return survived

# match satellite distribution & number by drawing subhalo closest
def match_rnum(df, sample, plotlabel=None,
                plotpath='figures/sampling/match_radial_dists/'):
    subs_c = df.copy()
    subs_c = subs_c[subs_c.r < 300]

    # get MW satellite distribution
    MC_dwarfs = np.load('data/sampling/'+sample+'_converted.npy')
    dists = np.median(MC_dwarfs[:,6,:], axis=1)
    errors = np.std(MC_dwarfs[:,6,:], axis=1)
    p = errors.argsort()
    errors = errors[p]
    dists = dists[p]

    # select subhalos
    subs = subs_c.copy()
    selected = []
    for i in range(len(dists)):
        dist = dists[i]
        error = errors[i]
        ii = np.argmin(np.abs(subs.r - dist).values)
        diff = dist - subs.iloc[ii].r
        index = subs.iloc[ii].name
        subs.drop([index], inplace=True)
        selected.append(index)
    assert len(selected) == len(set(selected))
    survived = subs_c.loc[selected]

    # plot if wanted
    if plotlabel is not None:
        plotbins = np.arange(301, step=1)
        k = dict(histtype='step', cumulative=True, bins=plotbins, lw=2)
        w = np.ones_like(survived.r)/float(len(survived.r))
        plt.hist(survived.r, weights=w, label='corrected sim', **k)
        w = np.ones_like(dists)/float(len(dists))
        plt.hist(dists, weights=w, label='MW sats', **k)
        w = np.ones_like(subs_c.r)/float(len(subs_c.r))
        plt.hist(subs_c.r, weights=w, label='original sim', **k)
        plt.legend(loc='upper left')
        plt.savefig(plotpath+plotlabel+'.png', bbox_inches='tight')
        plt.close()

    return survived

# variable sigma
def sigma(r, sigma0, r0, alpha):
    return sigma0*(1+(r/r0))**-alpha
