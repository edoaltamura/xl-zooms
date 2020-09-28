import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from typing import List


def halo_mass_function(
        vr_file: str = None,
        boxsize: float = None,
        n_bins: int = 15,
        bin_bounds: List[float] = [13., 16.],
        out_dir: str = None
):
    with h5.File(vr_file, 'r') as catalogue:
        M200c = catalogue['/Mass_200crit'][:] * 1.e10

    bins = 10 ** (np.linspace(bin_bounds[0], bin_bounds[1], n_bins))
    hist, edges = np.histogram(M200c, bins=bins)

    fig, ax = plt.subplots()
    ax.set_xlim(bin_bounds)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.step(M200c, hist, where='mid')
    plt.savefig(f"{out_dir}/halo_mass_function.png")
    plt.show()
    fig.close()

    volMpc3 = boxsize ** 3
    hmf_rescaled = hist / (volMpc3 * np.log10(bins))

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(bins, hmf_rescaled)
    plt.savefig(f"{out_dir}/halo_mass_function_rescaled.png")
    plt.show()
    fig.close()


halo_mass_function(
    boxsize=300.,
    vr_file="/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/stf_swiftdm_3dfof_subhalo_0036/stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0",
    out_dir='~'
)
