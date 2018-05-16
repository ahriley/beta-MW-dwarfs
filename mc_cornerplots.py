import numpy as np
import corner
import yaml
import matplotlib.pyplot as plt
import utils as u

# samples = np.load('data/mcmc/sampling_fritz.npy')
samples = np.load('data/mcmc/sampling_converted_fritz.npy')

# get dwarf names
dwarf_file = 'data/dwarfs/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())

samples = samples[:,:,9:12]
samples = np.swapaxes(samples,0,1)
ndim = samples.shape[-1]

sats = u.load_satellites('data/dwarfs/fritz_cart.csv')

# make corner plot for each dwarf
for name, sample in zip(names, samples):
    figure = corner.corner(sample,labels=[r"$v_r$",
                            r"$v_\theta$", r"$v_\phi$"],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,title_kwargs={"fontsize": 12})

    """
    # input values -- heliocentric coords
    d = dwarfs[name]
    input = [d['mu_alpha'], d['mu_delta'], d['vel_los'], d['Distance']]
    """
    # input values -- cartesian coords
    d = sats.loc[name]
    input = [d['v_r'], d['v_theta'], d['v_phi']]

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(input[i], color="r")

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(input[xi], color="r")
            ax.axhline(input[yi], color="r")
            ax.plot(input[xi], input[yi], "sr")

    figure.savefig('figures/samples_cart_vel/'+name+'.png')
    plt.close(figure)
    print("Plotted for "+name)
