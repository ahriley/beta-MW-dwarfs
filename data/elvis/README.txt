Each file contains the redshift zero catalog of resolved halos within the 
uncontaminated volume (for isolated systems, this is defined as a sphere centered
on the host with a radius equal to the distance to the nearest low resolution 
particle; for the paired systems, it is the union of two such spheres centered on
each giant).  

Included are halos with present day Vmax > 8 km/s, the resolution limit of the 
ELVIS Suite (the HiRes files include halos with Vmax > 3 km/s, though the 
resolution limit is not as well defined), that have centers within the
uncontaminated volumes (that is, that are closer to the host center than the 
nearest low resolution particle).  

The isolated systems have been shifted such that the host center is at the center 
of the box, (25/h, 25/h, 25/h) Mpc.  Similarly, the paired systems have had their
geometric centers placed at the box center.

Each file begins with listing of the columns and their units, then lists the 
properties of the host(s), then lists all the remaining objects that meet the 
above cuts.  The columns are listed in detail below.  All positions are in Mpc, 
all radii are in kpc, all velocities are in km/s, and all masses are in Msun.

1:  ID of the halo
2-4:  X, Y, and Z position of the halo (Mpc)
5-7:  Velocity of the halo in the X, Y, and Z directions (km/s)
8:  The z = 0 peak of the circular velocity, Vmax (km/s)
9:  The peak of the circular velocity at a_peak, Vpeak (km/s)
10:  The current virial mass (for centrals; defined below) or bound mass (for 
     	 satellites) as defined by rockstar, Mv (Msun)
11:  The peak virial mass over all timesteps, Mpeak (Msun)
12:  The current virial radius (for centrals) or subhalo radius defined by 
     	 rockstar, Rv (kpc)
13:  The radius at which the z = 0 circular velocity curve peaks, Rmax (kpc)
14:  The latest scale factor at which Mv = Mpeak, a_peak
15:  The stellar mass implied from the modified abundance matching relationship 
     	 presented in Garrison-Kimmel+2013, Mstar_preferred (Msun)
16:  The stellar mass implied from the abundance matching relationship presented
     	 in Behroozi+2013, Mstar_Behroozi (Msun)
17:  Number of particles in the halo, npart
18:  The ID of the parent halo (set to -1 for central), PID
19:  The ID of the topmost halo in the hierarcy for non-centrals, upID


The ELVIS simulations were run in a WMAP-7 cosmology (h = 0.71, sigma_8 = 0.801, 
Omega_M = 0.266, Omega_L = 0.734, n_s = 0.963).  Virial masses refer to the mass 
within a sphere that corresponds to an overdensity of 97 relative to the critical 
density.  All data in these catalogs are as found by the Rockstar halo finder 
(Behroozi+2013, http://code.google.com/p/rockstar/) and consistent-trees 
(Behroozi+2013, https://code.google.com/p/consistent-trees/); data from the Amiga 
Halo Finder are available upon request.

Please cite Garrison-Kimmel+2013
(http://adsabs.harvard.edu/abs/2013arXiv1310.6746G) if you use any of the data in 
these catalogs.
