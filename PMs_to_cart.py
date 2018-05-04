import numpy as np
import pandas as pd
import yaml
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord

dwarf_file = 'data/dwarfs.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)

x, y, z, vx, vy, vz = [np.empty(0) for i in range(6)]
names = []

for name in dwarfs.keys():
    dsph = dwarfs[name]
    sc = SkyCoord(ra=dsph['RA']*u.degree, dec=dsph['DEC']*u.degree,
                    distance=dsph['Distance']*u.kpc,
                    pm_ra_cosdec=dsph['mu_alpha']*u.mas/u.yr,
                    pm_dec=dsph['mu_delta']*u.mas/u.yr,
                    radial_velocity=dsph['vel_los']*u.km/u.s, frame='icrs',)
    sc = sc.transform_to(coord.Galactocentric)

    names.append(name)
    x = np.append(x, sc.x.to(u.Mpc).value)
    y = np.append(y, sc.y.to(u.Mpc).value)
    z = np.append(z, sc.z.to(u.Mpc).value)
    vx = np.append(vx, sc.v_x.to(u.km/u.s).value)
    vy = np.append(vy, sc.v_y.to(u.km/u.s).value)
    vz = np.append(vz, sc.v_z.to(u.km/u.s).value)

df = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
df = pd.DataFrame(df, index=names)
# df.to_csv('data/dwarfs_cartesian.csv')
# pd.read_csv('data/dwarfs_cartesian.csv', index_col=0)
