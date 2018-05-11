import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm

# read in dwarfs yaml
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

# plot the dwarfs as Gaussians in heliocentric distance
x = np.arange(0,50,0.5)
for name in names:
    d = dwarfs[name]
    rv = norm(loc=d['Distance'], scale=d['Distance_error'])
    print(name+" "+str(d['Distance'])+" "+str(d['Distance_error']))
    plt.plot(x, rv.pdf(x))
plt.title("Gaussian distribution of dwarfs")
plt.xlabel("Galactocentric distance [kpc]")
plt.ylabel("P(r)")
plt.savefig('dwarfs_r_gaussian_50kpc.png', bbox_inches='tight')
