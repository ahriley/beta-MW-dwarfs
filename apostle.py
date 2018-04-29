import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

Mpc_to_km = 3.086*10**19
km_to_kpc = 10**3/Mpc_to_km

sim = 'V1_HR_fix'
subs = load_apostle(sim)
haloIDs = list(set(subs.hostID))
assert len(haloIDs) == 2
subs, halos = subs.drop(haloIDs), subs.loc[haloIDs]

halos = halos.sort_values('M_star', ascending=False)
And_id = int(halos.iloc[0].hostID)
MW_id = int(halos.iloc[1].hostID)

subs = center_on_hosts(hosts=halos, subs=subs)
subs.x, subs.y, subs.z = subs.x*Mpc_to_km, subs.y*Mpc_to_km, subs.z*Mpc_to_km
subs = compute_spherical_hostcentric_sameunits(df=subs)
subs.x, subs.y, subs.z = subs.x*km_to_kpc, subs.y*km_to_kpc, subs.z*km_to_kpc
subs.r = subs.r*km_to_kpc
