import numpy as np
import pandas as pd
import glob

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

    df2 = df.copy()
    df2['r'], df2['theta'], df2['phi'] = r, theta, phi
    df2['v_r'], df2['v_theta'], df2['v_phi'] = v_r, v_theta, v_phi
    df2['v_t'] = np.sqrt(v_theta**2 + v_phi**2)

    return df2

def list_of_sims(suite):
    if suite == 'elvis':
        files = glob.glob('data/elvis/*.txt')
        files.remove('data/elvis/README.txt')
        return [f[11:-4] for f in files]
    elif suite == 'apostle':
        files = glob.glob('data/apostle/*.pkl')
        return [f[13:-9] for f in files]
    else:
        raise ValueError("suite must be 'elvis' or 'apostle'")

def load_apostle(sim):
    filename = 'data/apostle/'+sim+'_subs.pkl'
    df = pd.read_pickle(filename)
    return df.drop_duplicates()

def load_elvis(sim):
    filename = 'data/elvis/'+sim+'.txt'

    # read in the data
    with open(filename) as f:
        ID, x, y, z, vx, vy, vz, mvir, mstar = [[] for i in range(9)]
        for line in f:
            if line[0] == '#':
                continue
            items = line.split()
            ID.append(int(items[0]))
            x.append(float(items[1]))
            y.append(float(items[2]))
            z.append(float(items[3]))
            vx.append(float(items[4]))
            vy.append(float(items[5]))
            vz.append(float(items[6]))
            mvir.append(float(items[9]))
            mstar.append(float(items[14]))
    df = {'M_dm': mvir, 'M_star': mstar, 'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz}
    df = pd.DataFrame(df, index=ID)
    df.index.name = 'ID'

    # main halo association just based on distance
    host1 = df.iloc[0]
    host2 = df.iloc[1]
    df['d1'] = np.sqrt((df.x-host1.x)**2+(df.y-host1.y)**2+(df.z-host1.z)**2)
    df['d2'] = np.sqrt((df.x-host2.x)**2+(df.y-host2.y)**2+(df.z-host2.z)**2)
    df['hostID'] = -1
    df.loc[df['d1'] < df['d2'], 'hostID'] = host1.name
    df.loc[df['d1'] > df['d2'], 'hostID'] = host2.name
    df.drop(columns=['d1', 'd2'], inplace=True)

    return df
