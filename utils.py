import numpy as np
import pandas as pd
import glob

Mpc2km = 3.086*10**19
km2kpc = 10**3/Mpc2km

ELVIS_DIR = '/Users/alexanderriley/Desktop/elvis/'

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

def compute_spherical_hostcentric_sameunits(df):
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
    assert np.isclose(cart, sphere).all()
    assert np.isclose(cart, sphere2).all()

    df2 = df.copy()
    df2['r'], df2['theta'], df2['phi'] = r, theta, phi
    df2['v_r'], df2['v_theta'], df2['v_phi'] = v_r, v_theta, v_phi
    df2['v_t'] = v_t

    return df2

def list_of_sims(sim):
    if sim != 'elvis':
        raise NotImplementedErorr("Simulation should be 'elvis' for now")
    files_all = glob.glob(ELVIS_DIR+'*.txt')
    files = []
    for file in files_all:
        if '&' in file:
            files.append(file)
    return [f[len(ELVIS_DIR):-4] for f in files]

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
            'M_dm': mvir, 'M_star': mstar, 'Mpeak': mpeak, 'Mstar_b': mstar_b,
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
