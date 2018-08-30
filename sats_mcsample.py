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
from astropy.coordinates.representation import CartesianDifferential as CD

n = 10000
plot = True

dwarf_file = 'data/dwarf_props.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs_c = yaml.load(f)

studies = ['helmi', 'simon', 'fritz', 'kallivayalil', 'massari', 'pace']
for study in studies:
    print(study)
    if plot:
        pltpth = 'figures/sampling/'+study
        kwargs = {'parents': True, 'exist_ok': True}
        pathlib.Path(pltpth+'/heliocentric').mkdir(**kwargs)
        pathlib.Path(pltpth+'/galactocentric').mkdir(**kwargs)

    # read in dwarfs yaml
    dwarf_file = 'data/'+study+'.yaml'
    with open(dwarf_file, 'r') as f:
        dwarfs = yaml.load(f)
        names = dwarfs.keys()

    # get N galaxy parameters using errors
    galpars = np.zeros((n,5))
    galpars[:,0] = np.random.normal(8.2, 0.1, n)
    galpars[:,1] = np.random.normal(10, 1, n)
    galpars[:,2] = np.random.normal(248, 3, n)
    galpars[:,3] = np.random.normal(7, 0.5, n)
    galpars[:,4] = np.random.normal(25, 5, n)

    # for each dwarf, compute a sample in heliocentric
    positions = []
    coords_converted = []
    for name in names:
        d = dwarfs[name]
        d_c = dwarfs_c[name]
        pos = np.zeros((n,6))

        means = np.array([d['mu_alpha'], d['mu_delta']])
        cov = [[d['mu_alpha_error']**2,d['correlation_mu_mu']*\
                    d['mu_alpha_error']*d['mu_delta_error']],
                [d['correlation_mu_mu']*d['mu_alpha_error']*\
                    d['mu_delta_error'], d['mu_delta_error']**2]]
        cov = np.array(cov)
        pos[:,0:2] = np.random.multivariate_normal(mean=means, cov=cov, size=n)
        pos[:,2] = np.random.normal(d_c['vel_los'],d_c['vel_los_error'],n)
        pos[:,3] = np.random.normal(d_c['distance'],d_c['distance_error'],n)
        pos[:,4] = np.ones(n)*d_c['ra']
        pos[:,5] = np.ones(n)*d_c['dec']

        positions.append(np.swapaxes(pos, axis1=0, axis2=1))
    positions = np.array(positions)

    # convert that sample to galactocentric (this is the slow bit)
    cartcoords = np.zeros_like(positions)
    for i in range(n):
        if ((i/n)*100)%10 == 0 and i != 0:
            print('{0:.1f}%'.format(i/n*100))

        sc = SkyCoord(ra=positions[:,4,i]*u.degree,
                        dec=positions[:,5,i]*u.degree,
                        distance=positions[:,3,i]*u.kpc,
                        pm_ra_cosdec=positions[:,0,i]*u.mas/u.yr,
                        pm_dec=positions[:,1,i]*u.mas/u.yr,
                        radial_velocity=positions[:,2,i]*u.km/u.s,
                        frame='icrs')

        frame = coord.Galactocentric(galcen_distance=galpars[i,0]*u.kpc,
                        galcen_v_sun=CD(galpars[i,1:4]*u.km/u.s),
                        z_sun=galpars[i,4]*u.pc)

        sc = sc.transform_to(frame)
        assert frame.galcen_distance == sc.galcen_distance
        cartcoords[:,0,i] = sc.x.to(u.kpc).value
        cartcoords[:,1,i] = sc.y.to(u.kpc).value
        cartcoords[:,2,i] = sc.z.to(u.kpc).value
        cartcoords[:,3,i] = sc.v_x.to(u.km/u.s).value
        cartcoords[:,4,i] = sc.v_y.to(u.km/u.s).value
        cartcoords[:,5,i] = sc.v_z.to(u.km/u.s).value

    # convert cartesian to spherical, make corner plots
    sphcoords = np.zeros((len(names), 13, n))
    for s, i in zip(names, range(len(names))):
        df = {'x': cartcoords[i,0,:], 'y': cartcoords[i,1,:],
                'z': cartcoords[i,2,:], 'vx': cartcoords[i,3,:],
                'vy': cartcoords[i,4,:], 'vz': cartcoords[i,5,:]}
        df = pd.DataFrame(df)
        df = utils.compute_spherical_hostcentric(df=df)
        sphcoords[i] = np.swapaxes(df.values, axis1=0, axis2=1)

        if plot:
            kwargs = {'quantiles': [0.16, 0.5, 0.84], 'show_titles': True,
                        'title_kwargs': {'fontsize': 12}}
            labels = [r"$\mu_{\alpha^*}$", r"$\mu_\delta$", r"$v_\odot$",
                        r"$d_\odot$"]
            fig = corner.corner(positions[i][:4].T, labels=labels, **kwargs)
            fig.savefig(pltpth+'/heliocentric/'+s+'.png',bbox_inches='tight')
            plt.close()

            labels = [r"$r$", r"$\theta$", r"$\phi$", r"$v_r$", r"$v_\theta$",
                        r"$v_\phi$"]
            fig = corner.corner(sphcoords[i][6:12].T, labels=labels, **kwargs)
            fig.savefig(pltpth+'/galactocentric/'+s+'.png',bbox_inches='tight')
            plt.close()

    assert positions.shape == (len(names), 6, n)
    assert sphcoords.shape == (len(names), 13, n)

    # ORDER: mu_alpha, mu_delta, vel_los, dist
    np.save('data/sampling/'+study+'_helio', positions[:,:4])

    # ORDER: vx vy vz x y z r theta phi v_r v_theta v_phi v_t
    np.save('data/sampling/'+study+'_galacto', sphcoords)
