import numpy as np
import pandas as pd

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
