import pyatomdb, numpy
from matplotlib import pyplot as plt

import sys

chandra = '/cosma/home/dp004/dc-alta2/data7/xl-zooms/analysis/xray-analysis/pyatomdb/pyatomdb/examples'


def calc_power(Zlist, cie, Tlist):
    res = {}
    res['power'] = {}
    res['temperature'] = []
    cie.set_abund(numpy.arange(1, 31), 0.0)
    kTlist = Tlist * pyatomdb.const.KBOLTZ
    en = (cie.ebins_out[1:] + cie.ebins_out[:-1]) / 2
    for i, kT in enumerate(kTlist):

        print("Doing temperature iteration %i of %i" % (i, len(kTlist)))
        T = Tlist[i]

        res['temperature'].append(T)
        res['power'][i] = {}

        for Z in Zlist:

            if Z == 0:
                # This is the electron-electron bremstrahlung component alone
                # set all abundances to 1 (I need a full census of electrons in the plasma for e-e brems)
                cie.set_abund(Zlist[1:], 1.0)
                cie.set_eebrems(True)
                spec = cie.return_spectrum(kT, dolines=False, docont=False, dopseudo=False)

            else:
                # This is everything else, element by element.
                # turn off all the elements
                cie.set_abund(Zlist[1:], 0.0)
                # turn back on this element
                cie.set_abund(Z, 1.0)
                # turn off e-e bremsstrahlung (avoid double counting)
                cie.set_eebrems(False)
                spec = cie.return_spectrum(kT)

            res['power'][i][Z] = sum(spec * en)

    return res


if __name__ == '__main__':
    # Elements to include
    # ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    Zlist = [0, 1, 2, 6, 7, 8, 10, 12, 13, 14, 26]
    Elo = 0.5  # keV
    Ehi = 2.0
    kT = 8.  # temperature in keV

    energy_bins = numpy.logspace(numpy.log10(Elo), numpy.log10(Ehi), 10000)

    # declare the Collisional Ionization Equilibrium session
    sess = pyatomdb.spectrum.CIESession()
    sess.set_abund(Zlist[1:], 1.0)
    sess.set_eebrems(True)
    sess.set_broadening(True, velocity_broadening=400.)  # velocity in km/s
    # sess.set_response(energy_bins, raw=True)
    # now repeat the process with a real response
    sess.set_response(chandra + '/aciss_meg1_cy22.grmf', arf=chandra + '/aciss_meg1_cy22.garf')
    spec = sess.return_spectrum(kT)
    spec = numpy.append(0, spec)

    # Returned spectrum has units of photons cm^5 s^-1 bin^-1
    fig, ax = plt.subplots()
    ax.plot(sess.ebins_out, spec, drawstyle='steps')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Intensity (ph cm$^5$ s$^{-1}$ bin$^{-1}$)')
    plt.show()
