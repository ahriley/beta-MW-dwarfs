import numpy as np
import pandas as pd
import yaml
import corner
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
import utils
import pickle

study = 'magclouds'
plot = False
n = 10000
edges = np.arange(0,250+1,50)
rvals = (edges[1:] + edges[:-1]) / 2
head = str(edges)

# read in dwarfs yaml
dwarf_file = 'data/dwarfs/'+study+'.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

# for each dwarf, compute a sample
positions = []
for name in names:
    d = dwarfs[name]
    pos = np.zeros((n,4))

    means = np.array([d['mu_alpha'], d['mu_delta']])
    cov = np.array([[d['mu_alpha_error']**2, d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error']],
                    [d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error'], d['mu_delta_error']**2]])
    pos[:,0:2] = np.random.multivariate_normal(mean=means, cov=cov, size=n)
    pos[:,2] = np.random.normal(d['vel_los'],d['vel_los_error'],n)
    pos[:,3] = np.random.normal(d['Distance'],d['Distance_error'],n)

    if plot:
        figure = corner.corner(pos,labels=[r"$\mu_\alpha$",
                                r"$\mu_\delta$", r"$v_{LOS}$", r"$d$"],
                                quantiles=[0.16, 0.5, 0.84],title_fmt='.3f',
                                show_titles=True,title_kwargs={"fontsize": 12})
        figure.savefig('figures/mcmc/'+name+'.png')
    positions.append(pos)
print("Generated MCMC samplings")
positions = np.array(positions)
assert positions.shape == (len(names), n, 4)

# for each sample, compute beta
beta_profile = []
coords_converted = []
status = 0.01
print("Converting SkyCoords, binning by r, computing beta")
for i in range(n):
    if i/n >= status:
        print("{0:.0f}%".format(i/n * 100))
        status += 0.01

    pos_i = positions[:,i,]

	# convert each dwarf from celestial to cartesian
    x,y,z,vx,vy,vz = [[] for i in range(6)]
    for name, pos in zip(names, pos_i):
        dsph = dwarfs[name]
        sc = SkyCoord(ra=dsph['RA']*u.degree, dec=dsph['DEC']*u.degree,
                        distance=pos[3]*u.kpc,
                        pm_ra_cosdec=pos[0]*u.mas/u.yr,
                        pm_dec=pos[1]*u.mas/u.yr,
                        radial_velocity=pos[2]*u.km/u.s, frame='icrs')
        sc = sc.transform_to(coord.Galactocentric)

        x = np.append(x, sc.x.to(u.Mpc).value)
        y = np.append(y, sc.y.to(u.Mpc).value)
        z = np.append(z, sc.z.to(u.Mpc).value)
        vx = np.append(vx, sc.v_x.to(u.km/u.s).value)
        vy = np.append(vy, sc.v_y.to(u.km/u.s).value)
        vz = np.append(vz, sc.v_z.to(u.km/u.s).value)

    df = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
    df = pd.DataFrame(df, index=names)

    # coordinate conversion, save for later
    df.x, df.y, df.z = df.x*utils.Mpc2km, df.y*utils.Mpc2km, df.z*utils.Mpc2km
    df = utils.compute_spherical_hostcentric_sameunits(df=df)
    df.x, df.y, df.z = df.x*utils.km2kpc, df.y*utils.km2kpc, df.z*utils.km2kpc
    df.r = df.r*utils.km2kpc
    coords_converted.append(df.values)
    """
	# compute beta in different radial bins
    df = df[(df.r < 250) & (df.index != 'Cra I')]
    cut = pd.cut(df['r'], bins=edges, labels=False)
    beta_df = np.empty(0)
    for i in range(len(rvals)):
        beta_df = np.append(beta_df, utils.beta(df[cut == i]))
    beta_profile.append(beta_df)
beta_profile = np.array(beta_profile)
"""
coords_converted = np.array(coords_converted)

# save both the MC sampling and the computed profiles
# np.savetxt('data/mcmc/beta_profile_'+study+'.txt', beta_profile, header=head)

# SHAPE: 39 dwarfs, 10000 samplings, 4 params
# ORDER: mu_alpha, mu_delta, vel_los, dist
np.save('data/mcmc/sampling_'+study, positions, allow_pickle=False)

# SHAPE: 10000 samplings, 39 dwarfs, 13 params
# ORDER: vx vy vz x y z r theta phi v_r v_theta v_phi v_t
np.save('data/mcmc/sampling_converted_'+study, coords_converted, allow_pickle=False)
