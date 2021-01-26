import pyatomdb, numpy, os, pylab

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
                # turn on e-e bremsstrahlung
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

    ############################
    #### ADJUST THINGS HERE ####
    ############################

    # Elements to include
    # ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    Zlist = [0, 1, 2, 6, 7, 8, 10, 12, 13, 14, 26]
    # Note that for this purpose, Z=0 is the electron-electron bremsstrahlung
    # continuum. This is not a general AtomDB convention, just what I've done here
    # to make this work.

    # specify photon energy range you want to integrate over (min = 0.001keV, max=100keV)
    Elo = 0.001  # keV
    Ehi = 100.0  #

    # temperatures at which to calculate curve (K)
    Tlist = numpy.logspace(4, 9, 51)

    ################################
    #### END ADJUST THINGS HERE ####
    ################################
    cie = pyatomdb.spectrum.CIESession()
    ebins = numpy.linspace(Elo, Ehi, 10001)
    cie.set_response(ebins, raw=True)
    cie.set_eebrems(True)
    k = calc_power(Zlist, cie, Tlist)

    fig = pylab.figure()
    fig.show()
    ax = fig.add_subplot(111)
    ax.loglog(k['temperature'], k['totpower'] * pyatomdb.const.ERG_KEV)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Radiated Power (erg cm$^3$ s$^{-1}$)')
    ax.set_xlim(min(Tlist), max(Tlist))
    pylab.draw()
    # fig.savefig('calc_power_examples_1_1.svg')
