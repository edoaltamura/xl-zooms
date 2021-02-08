import pyatomdb, numpy, os
from matplotlib import pyplot as plt

"""
This code is an example of generating a cooling curve: the total power
radiated in keV cm3 s-1 by each element at each temperature. It will
generate a text file with the emission per element at each temperature
from 1e4 to 1e9K.

This is similar to the atomdb.lorentz_power function, but with a few
normalizations removed to run a little quicker.

Note that the Anders and Grevesse (1989) abundances are built in to
this. These can be looked up using atomdb.get_abundance(abundset='AG89'),
or the 'angr' column of the table at
https://heasarc.nasa.gov/xanadu/xspec/xspec11/manual/node33.html#SECTION00631000000000000000
"""


def calc_power(Zlist, cie, Tlist):
    """
  Zlist : [int]
    List of element nuclear charges
  cie : CIESession
    The CIEsession object with all the relevant data predefined.
  Tlist : array(float)
    The temperatures at which to calculate the power (in K)
  """

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
            # if Z = 1, do the eebrems (only want to calculate this once)
            #      if Z == 1:
            #        cie.set_eebrems(True)
            #      else:
            #        cie.set_eebrems(False)

            # get spectrum in ph cm3 s-1
            # spec = cie.return_spectrum(kT)

            # convert to keV cm3 s-1, sum
            res['power'][i][Z] = sum(spec * en)

    return res


if __name__ == '__main__':

    # ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    Zlist = [0, 1, 2, 6, 7, 8, 10, 12, 13, 14, 26]
    Elo = 0.001  # keV
    Ehi = 100.0

    # temperatures at which to calculate curve (K)
    Tlist = numpy.logspace(4, 10, 101)

    # set up the spectrum
    cie = pyatomdb.spectrum.CIESession()
    ebins = numpy.linspace(Elo, Ehi, 10001)
    cie.set_response(ebins, raw=True)
    cie.set_eebrems(True)
    k = calc_power(Zlist, cie, Tlist)

    s = '# Temperature log10(K)'
    for i in range(len(Zlist)):
        s += ' %12i' % (Zlist[i])

    k['totpower'] = numpy.zeros(len(k['temperature']))

    for i in range(len(k['temperature'])):
        s = '%22e' % (numpy.log10(k['temperature'][i]))
        for Z in Zlist:
            s += ' %12e' % (k['power'][i][Z])
            k['totpower'][i] += k['power'][i][Z]

    fig, ax = plt.subplots()
    ax.plot(k['temperature'], k['totpower'] * pyatomdb.const.ERG_KEV)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Radiated Power (erg cm$^3$ s$^{-1}$)')
    ax.set_xlim(min(Tlist), max(Tlist))
    plt.show()
