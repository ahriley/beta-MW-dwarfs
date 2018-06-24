import numpy as np
import pandas as pd
import yaml
import corner
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
import utils
import pathlib
import matplotlib.pyplot as plt

n = 10000
plot = True

# centralized file for dwarf params
dwarf_file = 'data/dwarfs/dwarf_props.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs_c = yaml.load(f)

studies = ['helmi', 'simon', 'fritz', 'kallivayalil', 'massari']
for study in studies:
    if plot:
        pathlib.Path('figures/sampling/'+study+'/heliocentric').mkdir(parents=True, exist_ok=True)
        pathlib.Path('figures/sampling/'+study+'/galactocentric').mkdir(parents=True, exist_ok=True)

    # read in dwarfs yaml
    dwarf_file = 'data/dwarfs/'+study+'.yaml'
    with open(dwarf_file, 'r') as f:
        dwarfs = yaml.load(f)

    # for each dwarf, compute a sample
    positions = []
    coords_converted = []
    for name in dwarfs.keys():
        d = dwarfs[name]
        d_c = dwarfs_c[name]
        pos = np.zeros((n,4))

        means = np.array([d['mu_alpha'], d['mu_delta']])
        cov = [[d['mu_alpha_error']**2,d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error']],
                [d['correlation_mu_mu']*d['mu_alpha_error']*d['mu_delta_error'], d['mu_delta_error']**2]]
        cov = np.array(cov)
        pos[:,0:2] = np.random.multivariate_normal(mean=means, cov=cov, size=n)
        pos[:,2] = np.random.normal(d_c['vel_los'],d_c['vel_los_error'],n)
        pos[:,3] = np.random.normal(d_c['distance'],d_c['distance_error'],n)

        positions.append(np.swapaxes(pos, axis1=0, axis2=1))

        sc = SkyCoord(ra=d_c['ra']*u.degree, dec=d_c['dec']*u.degree,
                        distance=pos[:,3]*u.kpc,
                        pm_ra_cosdec=pos[:,0]*u.mas/u.yr,
                        pm_dec=pos[:,1]*u.mas/u.yr,
                        radial_velocity=pos[:,2]*u.km/u.s, frame='icrs')

        sc = sc.transform_to(coord.Galactocentric)

        x = sc.x.to(u.kpc).value
        y = sc.y.to(u.kpc).value
        z = sc.z.to(u.kpc).value
        vx = sc.v_x.to(u.km/u.s).value
        vy = sc.v_y.to(u.km/u.s).value
        vz = sc.v_z.to(u.km/u.s).value

        df = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
        df = pd.DataFrame(df)
        df = utils.compute_spherical_hostcentric(df=df)
        coords_converted.append(np.swapaxes(df.values, axis1=0, axis2=1))

        if plot:
            fig = corner.corner(pos, labels=[r"$\mu_{\alpha^*}$",
                                    r"$\mu_\delta$", r"$v_\odot$", r"$d_\odot$"],
                                    quantiles=[0.16, 0.5, 0.84],
                                    show_titles=True, title_kwargs={"fontsize": 12})
            fig.savefig('figures/sampling/'+study+'/heliocentric/'+name+'.png', bbox_inches='tight')
            plt.close()

            fig = corner.corner(df.values[:,6:12], labels=[r"$r$", r"$\theta$", r"$\phi$",
                                    r"$v_r$", r"$v_\theta$", r"$v_\phi$"],
                                    quantiles=[0.16, 0.5, 0.84],
                                    show_titles=True, title_kwargs={"fontsize": 12})
            fig.savefig('figures/sampling/'+study+'/galactocentric/'+name+'.png', bbox_inches='tight')
            plt.close()

    positions = np.array(positions)
    coords_converted = np.array(coords_converted)
    assert positions.shape == (len(dwarfs.keys()), 4, n)
    assert coords_converted.shape == (len(dwarfs.keys()), 13, n)

    # ORDER: mu_alpha, mu_delta, vel_los, dist
    np.save('data/sampling/'+study, positions)

    # ORDER: vx vy vz x y z r theta phi v_r v_theta v_phi v_t
    np.save('data/sampling/'+study+'_converted', coords_converted)
