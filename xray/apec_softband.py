import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares
import unyt

np.seterr(divide='ignore')
kb = 1.3807e-16       # Boltzmann's constant [erg/K]
mpc = 3.0857e24       # Megaparsec [cm]
erg2keV = 1.60215e-9  # erg --> keV conversion factor


def radial_bin(r_dist, weights, rmin=0.1, rmax=1.0, nbins=50):
    bins = np.logspace(np.log10(rmin), np.log10(rmax), num=nbins)
    binned = np.histogram(r_dist.v, bins, weights=weights.v)[0]
    return np.array(bins) * r_dist.units, np.array(binned) * weights.units


def locate(A, x):
    fdx = A.searchsorted(x)
    fdx = np.clip(fdx, 1, len(A) - 1)
    lt = A[fdx - 1]
    rt = A[fdx]
    fdx -= x - lt < rt - x
    return fdx


def Xe(zmet):
    if zmet > 10 ** 0.5:
        zmet = 10 ** 0.5
    elif zmet < 0.0:
        zmet = 0.0
    ztab = [0.0, 10 ** -1.5, 10 ** -1.0, 10 ** -0.5, 1.0, 10 ** 0.5]
    xetab = [1.128, 1.129, 1.131, 1.165, 1.209, 1.238]
    ifunc = interp1d(ztab, xetab)
    rtn = ifunc(zmet)
    if rtn < ztab[0]:
        rtn = xetab[0]
    if rtn > ztab[-1]:
        rtn = xetab[-1]
    return rtn


def Xi(zmet):
    if zmet > 10 ** 0.5:
        zmet = 10 ** 0.5
    elif zmet < 0.0:
        zmet = 0.0
    ztab = [0.0, 10 ** -1.5, 10 ** -1.0, 10 ** -0.5, 1.0, 10 ** 0.5]
    xitab = [1.064, 1.064, 1.064, 1.08, 1.099, 1.103]
    ifunc = interp1d(ztab, xitab)
    rtn = ifunc(zmet)
    if rtn < ztab[0]:
        rtn = xitab[0]
    if rtn > ztab[-1]:
        rtn = xitab[-1]
    return rtn


def wabs(E, col):
    EX = np.array([0.100, 0.284, 0.400, 0.532, 0.707, 0.867, 1.303, 1.840, 2.471, 3.210, 4.038, 7.111, 8.331, 10.000])
    C0 = np.array([17.30, 34.60, 78.10, 71.40, 95.50, 308.9, 120.6, 141.3, 202.7, 342.7, 352.2, 433.9, 629.0, 701.2])
    C1 = np.array([608.1, 267.9, 18.80, 66.80, 145.8, -380.6, 169.3, 146.8, 104.7, 18.70, 18.70, -2.40, 30.90, 25.20])
    C2 = np.array([-2150.0, -476.1, 4.3, -51.4, -61.1, 294.0, -47.7, -31.5, -17.0, 0.000, 0.000, 0.750, 0.000, 0.000])
    ii = locate(EX, E)
    E3 = E ** 3
    AXS = (C0[ii] + C1[ii] * E + C2[ii] * E ** 2)
    BGDIF = AXS / E3
    BGDIF = -BGDIF * 1.0e-4
    arg = col / 1.0e20 * BGDIF
    mabs = np.exp(arg)
    return mabs


def mu2(zmet):
    if zmet > 10 ** 0.5:
        zmet = 10 ** 0.5
    elif zmet < 0:
        zmet = 0
    Ztab = np.array([0, 10 ** -1.5, 0.1, 10 ** -0.5, 1, 10 ** 0.5])
    MUtab = np.array([0.965 / 1.6726, 0.966 / 1.6726, 0.968 / 1.6726, 0.998 / 1.6726, 1.034 / 1.6726, 1.062 / 1.6726])
    ifunc = interp1d(Ztab, MUtab)
    rtn = ifunc(zmet)
    if rtn < Ztab[0]:
        rtn = MUtab[0]
    if rtn > Ztab[-1]:
        rtn = MUtab[-1]
    return rtn


def calc_spectrum(data, R500c):
    # Read APEC look up table: 0.05-100.0keV
    APEC_spec = h5py.File(f'APEC_spectra_0.05-100.0keV.hdf5', 'r')
    temptab = APEC_spec['Log_Plasma_Temp'][:]
    energies = APEC_spec['Energies'][:]
    APECtab = {}
    APECtab['Hydrogen'] = APEC_spec['Hydrogen'][:]
    APECtab['Helium'] = APEC_spec['Helium'][:]
    APECtab['Carbon'] = APEC_spec['Carbon'][:]
    APECtab['Nitrogen'] = APEC_spec['Nitrogen'][:]
    APECtab['Oxygen'] = APEC_spec['Oxygen'][:]
    APECtab['Neon'] = APEC_spec['Neon'][:]
    APECtab['Magnesium'] = APEC_spec['Magnesium'][:]
    APECtab['Silicon'] = APEC_spec['Silicon'][:]
    APECtab['Sulphur'] = APEC_spec['Sulphur'][:]
    APECtab['Calcium'] = APEC_spec['Calcium'][:]
    APECtab['Iron'] = APEC_spec['Iron'][:]

    data_out = dict()
    data_out['SpecEngr'] = energies
    data_out['SpecTemp'] = temptab
    data_out['SpecAPEC'] = APECtab

    ne_nH = np.zeros(len(data.gas.element_mass_fractions.hydrogen.value)) + 1
    ni_nH = np.zeros(len(data.gas.element_mass_fractions.hydrogen.value)) + 1
    mu = np.zeros(len(data.gas.element_mass_fractions.hydrogen.value))

    # -- Sum element contributions
    # Hydrogen
    H = data.gas.element_mass_fractions.hydrogen.value
    mu += 1.0 / (1.0 + 1.0)
    lN_H_AG = 12.00
    # Helium
    He_H = data.gas.element_mass_fractions.helium.value / H
    ne_nH += (He_H) * (1.00794 / 4.002602) * (2.0 / 1.0)
    ni_nH += (He_H) * (1.00794 / 4.002602)
    mu += (He_H) / (1.0 + 2.0)
    AG_He = 10.99 - lN_H_AG
    He_H = 10.0 ** (np.log10(He_H * (1.00794 / 4.002602)) - AG_He)
    # Carbon
    C_H = data.gas.element_mass_fractions.carbon.value / H
    ne_nH += (C_H) * (1.00794 / 12.0107) * (6.0 / 1.0)
    ni_nH += (C_H) * (1.00794 / 12.0107)
    mu += (C_H) / (1.0 + 6.0)
    AG_C = 8.56 - lN_H_AG
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)
    # Nitrogen
    N_H = data.gas.element_mass_fractions.nitrogen.value / H
    ne_nH += (N_H) * (1.00794 / 14.0067) * (7.0 / 1.0)
    ni_nH += (N_H) * (1.00794 / 14.0067)
    mu += (N_H) / (1.0 + 7.0)
    AG_N = 8.05 - lN_H_AG
    N_H = 10.0 ** (np.log10(N_H * (1.00794 / 14.0067)) - AG_N)
    # Oxygen
    O_H = data.gas.element_mass_fractions.oxygen.value / H
    ne_nH += (O_H) * (1.00794 / 15.9994) * (8.0 / 1.0)
    ni_nH += (O_H) * (1.00794 / 15.9994)
    mu += (O_H) / (1.0 + 8.0)
    AG_O = 8.93 - lN_H_AG
    O_H = 10.0 ** (np.log10(O_H * (1.00794 / 15.9994)) - AG_O)
    # Neon
    Ne_H = data.gas.element_mass_fractions.neon.value / H
    ne_nH += (Ne_H) * (1.00794 / 20.1797) * (10.0 / 1.0)
    ni_nH += (Ne_H) * (1.00794 / 20.1797)
    mu += (Ne_H) / (1.0 + 10.0)
    AG_Ne = 8.09 - lN_H_AG
    Ne_H = 10.0 ** (np.log10(Ne_H * (1.00794 / 20.1797)) - AG_Ne)
    # Magnesium
    Mg_H = data.gas.element_mass_fractions.magnesium.value / H
    ne_nH += (Mg_H) * (1.00794 / 24.3050) * (12.0 / 1.0)
    ni_nH += (Mg_H) * (1.00794 / 24.3050)
    mu += (Mg_H) / (1.0 + 12.0)
    AG_Mg = 7.58 - lN_H_AG
    Mg_H = 10.0 ** (np.log10(Mg_H * (1.00794 / 24.3050)) - AG_Mg)
    # Silicon, Sulphur & Calcium
    Si_H = data.gas.element_mass_fractions.silicon.value / H
    Ca_Si = 0.0941736
    S_Si = 0.6054160
    ne_nH += Si_H * (1.00794 / 28.0855) * (14.0 / 1.0)
    ne_nH += (Si_H * Ca_Si) * (1.00794 / 40.078) * (20.0 / 1.0)
    ne_nH += (Si_H * S_Si) * (1.00794 / 32.065) * (16.0 / 1.0)
    ni_nH += Si_H * (1.00794 / 28.0855)
    ni_nH += (Si_H * Ca_Si) * (1.00794 / 40.078)
    ni_nH += (Si_H * S_Si) * (1.00794 / 32.065)
    mu += Si_H / (1.0 + 14.0)
    mu += (Si_H * Ca_Si) / (1.0 + 20.0)
    mu += (Si_H * S_Si) / (1.0 + 16.0)
    AG_Si = 7.55 - lN_H_AG
    AG_Ca = 6.36 - lN_H_AG
    AG_S = 7.21 - lN_H_AG
    Ca_H = 10.0 ** (np.log10((Ca_Si * Si_H) * (1.00794 / 40.078)) - AG_Ca)
    S_H = 10.0 ** (np.log10((S_Si * Si_H) * (1.00794 / 32.065)) - AG_S)
    Si_H = 10.0 ** (np.log10(Si_H * (1.00794 / 28.0855)) - AG_Si)
    # Iron
    Fe_H = data.gas.element_mass_fractions.iron.value / H
    ne_nH += (Fe_H) * (1.00794 / 55.845) * (26.0 / 1.0)
    ni_nH += (Fe_H) * (1.00794 / 55.845)
    mu += (Fe_H) / (1.0 + 26.0)
    AG_Fe = 7.67 - lN_H_AG
    Fe_H = 10.0 ** (np.log10(Fe_H * (1.00794 / 55.845)) - AG_Fe)

    # Emission measure & Y parameter
    EMM = data.gas.densities * data.gas.masses / ne_nH * (ne_nH / ((ne_nH + ni_nH) * mu * unyt.proton_mass)) ** 2.0
    Ypar = (unyt.thompson_cross_section / (511.0 * unyt.keV)) * unyt.boltzmann_constant * data.gas.temperatures * (
            data.gas.masses * 0.752 * ne_nH / unyt.proton_mass) / unyt.Mpc ** 2

    # Calculate spectrum
    r = data.gas.radial_distances
    mass = data.gas.masses
    temp = data.gas.temperatures
    iron = data.gas.element_mass_fractions.iron

    nbins = 25 + 1
    rm = np.log10(0.15 * R500c)
    rx = np.log10(1 * R500c)
    rbin = np.logspace(rm, rx, num=nbins, base=10.0)
    rcen = 10.0 ** (0.5 * np.log10(rbin[1:] * rbin[:-1]))
    vol = (4.0 / 3.0) * np.pi * ((rbin[1:] ** 3.0) - (rbin[:-1] ** 3.0)) * unyt.Mpc ** 3

    mpro = radial_bin(r, mass, rmin=np.min(rbin), rmax=np.max(rbin), nbins=nbins)[1]
    tpro = radial_bin(r, mass * temp, rmin=np.min(rbin), rmax=np.max(rbin), nbins=nbins)[1]
    zpro = radial_bin(r, mass * iron, rmin=np.min(rbin), rmax=np.max(rbin), nbins=nbins)[1]

    spectrum = np.zeros((len(rcen), len(energies)))

    for k in range(len(rcen)):
        print(f"Calculating spectrum in shell ({k + 1}/{len(rcen)})")
        idx = np.where((r > rbin[k]) & (r <= rbin[k + 1]))[0]
        if len(idx) <= 0:
            continue
        itemp = locate(temptab, np.log10(temp[idx]))
        for j in range(0, len(energies), 1):
            spectrum[k, j] += np.sum(
                EMM[idx] * (APECtab['Hydrogen'][itemp, j] +
                            APECtab['Helium'][itemp, j] * He_H[idx] +
                            APECtab['Carbon'][itemp, j] * C_H[idx] +
                            APECtab['Nitrogen'][itemp, j] * N_H[idx] +
                            APECtab['Oxygen'][itemp, j] * O_H[idx] +
                            APECtab['Neon'][itemp, j] * Ne_H[idx] +
                            APECtab['Magnesium'][itemp, j] * Mg_H[idx] +
                            APECtab['Silicon'][itemp, j] * Si_H[idx] +
                            APECtab['Sulphur'][itemp, j] * S_H[idx] +
                            APECtab['Calcium'][itemp, j] * Ca_H[idx] +
                            APECtab['Iron'][itemp, j] * Fe_H[idx])
            )
    del r, temp, He_H, C_H, N_H, O_H, Ne_H, Mg_H, Ca_H, S_H, Si_H, Fe_H, rm, rx

    data_out['Srho'] = (mpro / vol).to('g/cm**3').value
    data_out['Svol'] = vol.to('cm**3').value
    data_out['Stmp'] = ((tpro / mpro) * unyt.boltzmann_constant).to('keV').value
    data_out['Smet'] = (zpro / mpro) / 1.29e-3
    data_out['Rspec'] = rcen
    data_out['Spectrum'] = spectrum
    data_out['EMM'] = EMM
    data_out['Ypar'] = Ypar
    data_out['rho_crit'] = data.metadata.cosmology.critical_density(data.metadata.z).to('g/cm**3').value

    return data_out


def fit_spectrum(spectrum_data):

    chand_area = np.loadtxt('chandra_acis-i_.area')
    etmp = chand_area[:, 0]
    atmp = chand_area[:, 1]

    idxa = locate(etmp, spectrum_data['SpecEngr'])
    aeff = atmp[idxa]
    del idxa
    DL = 250.0 * mpc
    tint = 1.0e6
    mabs = wabs(spectrum_data['SpecEngr'], 2.0e20)
    spec_sfac = mabs * aeff * tint / (4.0 * np.pi * DL * DL) / erg2keV
    spectrum_data['SpecSFac'] = spec_sfac
    del chand_area, etmp, atmp, aeff, DL, tint, mabs

    nbins = len(spectrum_data['Spectrum'])
    temp = np.zeros(nbins)
    rho = np.zeros(nbins)
    zmet = np.zeros(nbins)
    xisq = np.zeros(nbins)
    for k in range(nbins):
        spectrum = spectrum_data['Spectrum'][k]

        if np.max(spectrum) <= 0.0:
            temp[k] = 0.0
            rho[k] = 0.0
            zmet[k] = 0.0
            continue

        vol = spectrum_data['Svol'][k]
        Tg = np.log10(spectrum_data['Stmp'][k] * (erg2keV / kb))
        Zg = spectrum_data['Smet'][k] * (1.29e-3 / 1.89e-3)
        Dg = (spectrum_data['Srho'][k] ** 2.0) * (vol / 1.0e66) * (
                ((Xe(Zg) / ((Xe(Zg) + Xi(Zg)) * mu2(Zg) * unyt.proton_mass)) ** 2.0) / Xe(Zg))

        print(f"Fitting spectrum for shell ({k + 1}/{nbins})")
        params, fitxi, spec_mod = specfit(
            spectrum, Tg, Dg, Zg,
            spectrum_data['SpecEngr'],
            spectrum_data['SpecSFac'],
            spectrum_data['SpecTemp'],
            spectrum_data['SpecAPEC']
        )
        temp[k] = params.x[0]
        rho[k] = params.x[1]
        zmet[k] = params.x[2]
        xisq[k] = fitxi
        del spectrum, vol, Tg, Dg, Zg, params, fitxi, spec_mod

    spectrum_data['Tspec'] = (kb / erg2keV) * (10.0 ** temp)

    rho_spec = np.zeros(len(rho))
    for k in range(len(rho)):
        rho_spec[k] = np.sqrt(
            (rho[k] * 1.0e66 / spectrum_data['Svol'][k]) / (((Xe(zmet[k]) /
            ((Xe(zmet[k]) + Xi(zmet[k])) * mu2(zmet[k]) * unyt.proton_mass)) ** 2.0) / Xe(zmet[k]))
        ) / spectrum_data['rho_crit']
    spectrum_data['RHOspec'] = rho_spec
    spectrum_data['Zspec'] = zmet * (1.89e-3 / 1.29e-3)
    spectrum_data['XIspec'] = xisq

    del nbins, rho, temp, zmet, xisq, rho_spec
    return


def specfit(spectrum, Tg, Dg, Zg, energies, spec_sfac, temptab, APECtab):
    # Calc photons per bin (1Ms)
    ppb = (spectrum / energies) * spec_sfac
    # Set initial guess & fit limits -- 6<log(T)<9, norm > 0, 0<Z/Zsun<10
    ini_guess = [Tg, Dg, Zg]
    lims = [(6.0, 9.0), (0.0, np.inf), (0.01, 10.0)]
    # Fit only bins with ppb > 0 and 0.5 < E < 10.0
    tdx = np.where((ppb > 0.0) & (energies > 0.5) & (energies < 10.0))[0]
    # Initial model spectrum fit
    ebins = np.arange(len(energies))

    coef = []
    coef.append(minimize(spec_model, ini_guess,
                         args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'),
                         method='L-BFGS-B', bounds=lims, tol=1.0e-4, options={'maxiter': 1000}))

    coef.append(minimize(spec_model, ini_guess,
                         args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'),
                         method='SLSQP', bounds=lims, tol=1.0e-4, options={'maxiter': 500}))

    coef.append(minimize(spec_model, ini_guess,
                         args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'),
                         method='Nelder-Mead', tol=1.0e-4, options={'maxiter': 1000}))
    if coef[-1].x[2] < 0.0:
        coef[-1].success = False

    try:
        coef.append(least_squares(spec_model, ini_guess,
                                  args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'),
                                  bounds=([6.0, 0.0, 0.0], [9.0, np.inf, 10.0]), ftol=1.0e-4, max_nfev=1000))
    except:
        pass

    fitcoef = None
    xisq = 1.0e100

    for y in coef:
        if y.success is False:
            continue
        spec_mod = spec_model(y.x, ppb[tdx], ebins[tdx], temptab, APECtab, energies, spec_sfac, tdx)
        Xsq_m = np.sum((spec_mod[tdx] - ppb[tdx]) ** 2.0 / ppb[tdx] ** 2.0) / (len(ebins[tdx]) - 3)
        if Xsq_m < xisq:
            fitcoef = y
            xisq = Xsq_m
        del Xsq_m

    if fitcoef is None:
        fitcoef = coef[0]
        fitcoef.x = ini_guess
        spec_mod = spec_model(fitcoef.x, ppb[tdx], ebins[tdx], temptab, APECtab, energies, spec_sfac, tdx)
        xisq = 1.0e100
    del coef
    model_flux = spec_mod * energies / spec_sfac
    del ppb, ini_guess, lims, tdx, ebins, spec_mod
    return fitcoef, xisq, model_flux


def spec_model(p0, x, y, temptab, APECtab, energies, spec_sfac, tdx, meth='Mod'):
    # Locate T in look up table
    T, D, Z = p0
    idx = locate(temptab, T)
    if idx < 0:
        idx = 0
    if idx > len(temptab) - 2:
        idx = len(temptab) - 2

    dlogT = temptab[1] - temptab[0]
    # -- Element contributions
    # Hydrogen
    m = (np.log10(APECtab['Hydrogen'][idx + 1]) - np.log10(APECtab['Hydrogen'][idx])) / dlogT
    b = np.log10(APECtab['Hydrogen'][idx]) - m * temptab[idx]
    H = 10.0 ** (m * T + b)
    # Helium
    m = (np.log10(APECtab['Helium'][idx + 1]) - np.log10(APECtab['Helium'][idx])) / dlogT
    b = np.log10(APECtab['Helium'][idx]) - m * temptab[idx]
    He = 10.0 ** (m * T + b)
    # Carbon
    m = (np.log10(APECtab['Carbon'][idx + 1]) - np.log10(APECtab['Carbon'][idx])) / dlogT
    b = np.log10(APECtab['Carbon'][idx]) - m * temptab[idx]
    C = 10.0 ** (m * T + b)
    # Nitrogen
    m = (np.log10(APECtab['Nitrogen'][idx + 1]) - np.log10(APECtab['Nitrogen'][idx])) / dlogT
    b = np.log10(APECtab['Nitrogen'][idx]) - m * temptab[idx]
    N = 10.0 ** (m * T + b)
    # Oxygen
    m = (np.log10(APECtab['Oxygen'][idx + 1]) - np.log10(APECtab['Oxygen'][idx])) / dlogT
    b = np.log10(APECtab['Oxygen'][idx]) - m * temptab[idx]
    O = 10.0 ** (m * T + b)
    # Neon
    m = (np.log10(APECtab['Neon'][idx + 1]) - np.log10(APECtab['Neon'][idx])) / dlogT
    b = np.log10(APECtab['Neon'][idx]) - m * temptab[idx]
    Ne = 10.0 ** (m * T + b)
    # Magnesium
    m = (np.log10(APECtab['Magnesium'][idx + 1]) - np.log10(APECtab['Magnesium'][idx])) / dlogT
    b = np.log10(APECtab['Magnesium'][idx]) - m * temptab[idx]
    Mg = 10.0 ** (m * T + b)
    # Silcon
    m = (np.log10(APECtab['Silicon'][idx + 1]) - np.log10(APECtab['Silicon'][idx])) / dlogT
    b = np.log10(APECtab['Silicon'][idx]) - m * temptab[idx]
    Si = 10.0 ** (m * T + b)
    # Sulphur
    m = (np.log10(APECtab['Sulphur'][idx + 1]) - np.log10(APECtab['Sulphur'][idx])) / dlogT
    b = np.log10(APECtab['Sulphur'][idx]) - m * temptab[idx]
    S = 10.0 ** (m * T + b)
    # Calcium
    m = (np.log10(APECtab['Calcium'][idx + 1]) - np.log10(APECtab['Calcium'][idx])) / dlogT
    b = np.log10(APECtab['Calcium'][idx]) - m * temptab[idx]
    Ca = 10.0 ** (m * T + b)
    # Iron
    m = (np.log10(APECtab['Iron'][idx + 1]) - np.log10(APECtab['Iron'][idx])) / dlogT
    b = np.log10(APECtab['Iron'][idx]) - m * temptab[idx]
    Fe = 10.0 ** (m * T + b)
    # -- Model spectrum
    mod = D * (H + He + Z * (C + N + O + Ne + Mg + Si + S + Ca + Fe))
    mod = 1.0e66 * (mod / energies) * spec_sfac
    # -- Return error or model
    if meth == 'Flg':
        return np.sum((np.log10(y) - np.log10(mod[tdx])) ** 2.0)
    elif meth == 'Fln':
        return np.sum((y - mod[tdx]) ** 2.0)
    else:
        return mod


def soft_band(data, pix):
    from scipy.io.idl import readsav

    APEC = readsav('APEC_0.5_2.0keV_interp.idl')

    inde = 0  # 0 - erg/s, 1 - photons
    indz = locate(APEC['redshift'], data.metadata.z)
    indT = locate(APEC["ltemp"], np.log10(data.gas.temperatures))

    ne_nH = np.zeros(len(data.gas.masses.value)) + 1
    ni_nH = np.zeros(len(data.gas.masses.value)) + 1
    mu = np.zeros(len(data.gas.masses.value))
    Lambda = np.zeros(len(data.gas.masses.value), dtype=np.float64)

    # --- Sum element contributions
    # Hydrogen
    H = data.gas.element_mass_fractions.hydrogen.value
    mu += 1.0 / (1.0 + 1.0)
    lN_H_AG = 12.00
    Lambda += APEC["Lambda_hydrogen"][indz, indT, inde]
    # Helium
    He_H = data.gas.element_mass_fractions.helium.value / H
    ne_nH += (He_H) * (1.00794 / 4.002602) * (2.0 / 1.0)
    ni_nH += (He_H) * (1.00794 / 4.002602)
    mu += (He_H) / (1.0 + 2.0)
    AG_He = 10.99 - lN_H_AG
    He_H = 10.0 ** (np.log10(He_H * (1.00794 / 4.002602)) - AG_He)
    Lambda += He_H * APEC["Lambda_helium"][indz, indT, inde]
    del He_H
    # Carbon
    C_H = data.gas.element_mass_fractions.carbon.value / H
    ne_nH += (C_H) * (1.00794 / 12.0107) * (6.0 / 1.0)
    ni_nH += (C_H) * (1.00794 / 12.0107)
    mu += (C_H) / (1.0 + 6.0)
    AG_C = 8.56 - lN_H_AG
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)
    Lambda += C_H * APEC["Lambda_carbon"][indz, indT, inde]
    # Nitrogen
    N_H = data.gas.element_mass_fractions.nitrogen.value / H
    ne_nH += (N_H) * (1.00794 / 14.0067) * (7.0 / 1.0)
    ni_nH += (N_H) * (1.00794 / 14.0067)
    mu += (N_H) / (1.0 + 7.0)
    AG_N = 8.05 - lN_H_AG
    N_H = 10.0 ** (np.log10(N_H * (1.00794 / 14.0067)) - AG_N)
    Lambda += N_H * APEC["Lambda_nitrogen"][indz, indT, inde]
    del N_H
    # Oxygen
    O_H = data.gas.element_mass_fractions.oxygen.value / H
    ne_nH += (O_H) * (1.00794 / 15.9994) * (8.0 / 1.0)
    ni_nH += (O_H) * (1.00794 / 15.9994)
    mu += (O_H) / (1.0 + 8.0)
    AG_O = 8.83 - lN_H_AG
    O_H = 10.0 ** (np.log10(O_H * (1.00794 / 15.9994)) - AG_O)
    Lambda += O_H * APEC["Lambda_oxygen"][indz, indT, inde]
    del O_H
    # Neon
    Ne_H = data.gas.element_mass_fractions.neon.value / H
    ne_nH += (Ne_H) * (1.00794 / 20.1797) * (10.0 / 1.0)
    ni_nH += (Ne_H) * (1.00794 / 20.1797)
    mu += (Ne_H) / (1.0 + 10.0)
    AG_Ne = 8.09 - lN_H_AG
    Ne_H = 10.0 ** (np.log10(Ne_H * (1.00794 / 20.1797)) - AG_Ne)
    Lambda += Ne_H * APEC["Lambda_neon"][indz, indT, inde]
    del Ne_H
    # Magnesium
    Mg_H = data.gas.element_mass_fractions.magnesium.value / H
    ne_nH += (Mg_H) * (1.00794 / 24.3050) * (12.0 / 1.0)
    ni_nH += (Mg_H) * (1.00794 / 24.3050)
    mu += (Mg_H) / (1.0 + 12.0)
    AG_Mg = 7.58 - lN_H_AG
    Mg_H = 10.0 ** (np.log10(Mg_H * (1.00794 / 24.3050)) - AG_Mg)
    Lambda += Mg_H * APEC["Lambda_magnesium"][indz, indT, inde]
    del Mg_H
    # Silicon, Sulphur & Calcium
    Si_H = data.gas.element_mass_fractions.silicon.value / H
    Ca_Si = 0.0941736
    S_Si = 0.6054160
    ne_nH += (Si_H) * (1.00794 / 28.0855) * (14.0 / 1.0)
    ne_nH += (Si_H * Ca_Si) * (1.00794 / 40.078) * (20.0 / 1.0)
    ne_nH += (Si_H * S_Si) * (1.00794 / 32.065) * (16.0 / 1.0)
    ni_nH += (Si_H) * (1.00794 / 28.0855)
    ni_nH += (Si_H * Ca_Si) * (1.00794 / 40.078)
    ni_nH += (Si_H * S_Si) * (1.00794 / 32.065)
    mu += (Si_H) / (1.0 + 14.0)
    mu += (Si_H * Ca_Si) / (1.0 + 20.0)
    mu += (Si_H * S_Si) / (1.0 + 16.0)
    AG_Si = 7.55 - lN_H_AG
    AG_Ca = 6.36 - lN_H_AG
    AG_S = 7.21 - lN_H_AG
    Ca_H = 10.0 ** (np.log10((Ca_Si * Si_H) * (1.00794 / 40.078)) - AG_Ca)
    S_H = 10.0 ** (np.log10((S_Si * Si_H) * (1.00794 / 32.065)) - AG_S)
    Si_H = 10.0 ** (np.log10(Si_H * (1.00794 / 28.0855)) - AG_Si)
    Lambda += Si_H * APEC["Lambda_silicon"][indz, indT, inde]
    Lambda += Ca_H * APEC["Lambda_calcium"][indz, indT, inde]
    Lambda += S_H * APEC["Lambda_sulphur"][indz, indT, inde]
    del Si_H, Ca_H, S_H
    # Iron
    Fe_H = data.gas.element_mass_fractions.iron.value / H
    ne_nH += (Fe_H) * (1.00794 / 55.845) * (26.0 / 1.0)
    ni_nH += (Fe_H) * (1.00794 / 55.845)
    mu += (Fe_H) / (1.0 + 26.0)
    AG_Fe = 7.67 - lN_H_AG
    Fe_H = 10.0 ** (np.log10(Fe_H * (1.00794 / 55.845)) - AG_Fe)
    Lambda += Fe_H * APEC["Lambda_iron"][indz, indT, inde]
    del H, Fe_H, indT

    Lambda = unyt.unyt_array(Lambda, 'erg/s*cm**3')

    # --- Calculate observables
    Lx = Lambda * (data.gas.densities * (
            ne_nH / ((ne_nH + ni_nH) * mu * unyt.proton_mass)) ** 2.0) * data.gas.masses / ne_nH
    Sx = Lx / (4.0 * np.pi * pix * pix) / ((180.0 * 60.0 / np.pi) ** 2)
    Ypix = (unyt.thompson_cross_section / (511.0 * unyt.keV)) * unyt.boltzmann_constant * data.gas.temperatures * (
            data.gas.masses / (mu * unyt.proton_mass)) * (ne_nH / (ne_nH + ni_nH)) / (pix * pix)

    print("Soft band LX", np.sum(Lx.in_cgs()))
    print("Soft band SX", np.sum(Sx.in_cgs()))
    print("Soft band YX", np.sum(Ypix.in_cgs()))


if __name__ == '__main__':
    import swiftsimio as sw
    import h5py as h5

    d = '/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_Ref/'


    def process_single_halo(
            path_to_snap: str,
            path_to_catalogue: str
    ):
        # Read in halo properties
        with h5.File(f'{path_to_catalogue}', 'r') as h5file:
            M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
            R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
            XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
            YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
            ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

        # Read in gas particles to compute the core-excised temperature
        mask = sw.mask(f'{path_to_snap}', spatial_only=False)
        region = [[XPotMin - 0.5 * R500c, XPotMin + 0.5 * R500c],
                  [YPotMin - 0.5 * R500c, YPotMin + 0.5 * R500c],
                  [ZPotMin - 0.5 * R500c, ZPotMin + 0.5 * R500c]]
        mask.constrain_spatial(region)
        mask.constrain_mask(
            "gas", "temperatures",
            1.e5 * mask.units.temperature,
            1.e10 * mask.units.temperature
        )
        print(f"M_500_crit: {M500c:.3E}")

        data = sw.load(f'{path_to_snap}', mask=mask)

        # Select hot gas within sphere and without core
        deltaX = data.gas.coordinates[:, 0] - XPotMin
        deltaY = data.gas.coordinates[:, 1] - YPotMin
        deltaZ = data.gas.coordinates[:, 2] - ZPotMin
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

        # Keep only particles inside 5 R500crit
        index = np.where((deltaR > 0.15 * R500c) & (deltaR < R500c))[0]
        data.gas.radial_distances = deltaR[index]
        data.gas.densities = data.gas.densities[index]
        data.gas.masses = data.gas.masses[index]
        data.gas.temperatures = data.gas.temperatures[index]

        data.gas.element_mass_fractions.hydrogen = data.gas.element_mass_fractions.hydrogen[index]
        data.gas.element_mass_fractions.helium = data.gas.element_mass_fractions.helium[index]
        data.gas.element_mass_fractions.carbon = data.gas.element_mass_fractions.carbon[index]
        data.gas.element_mass_fractions.nitrogen = data.gas.element_mass_fractions.nitrogen[index]
        data.gas.element_mass_fractions.oxygen = data.gas.element_mass_fractions.oxygen[index]
        data.gas.element_mass_fractions.neon = data.gas.element_mass_fractions.neon[index]
        data.gas.element_mass_fractions.magnesium = data.gas.element_mass_fractions.magnesium[index]
        data.gas.element_mass_fractions.silicon = data.gas.element_mass_fractions.silicon[index]
        data.gas.element_mass_fractions.iron = data.gas.element_mass_fractions.iron[index]

        return data, R500c


    data, R500c = process_single_halo(
        path_to_snap=d + 'snapshots/L0300N0564_VR18_-8res_Ref_2749.hdf5',
        path_to_catalogue=d + 'stf/L0300N0564_VR18_-8res_Ref_2749/L0300N0564_VR18_-8res_Ref_2749.properties',
    )

    soft_band(data, 1)
    spec = calc_spectrum(data, R500c)
    fit_spectrum(spec)
    print('Tspec', spec['Tspec'])
    print('RHOspec', spec['RHOspec'])
    print('Zspec', spec['Zspec'])
    print('XIspec', spec['XIspec'])
    print('EMM', np.sum(spec['EMM']))
    print('Ypar', np.sum(spec['Ypar']))
