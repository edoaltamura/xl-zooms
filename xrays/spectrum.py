def calc_spec(comm, NProcs, MyRank, x, data):
    import read
    import basics as bsc
    import h5py
    import numpy as np
    from mpi4py import MPI
    np.seterr(invalid='ignore', divide='ignore')

    ck = read.get_data(comm, NProcs, MyRank, x, 'Spectrum', data, [])

    # Read APEC look up table: 0.05-100.0keV
    APEC_spec = h5py.File('/cosma/home/dp004/dc-hens1/apec_tables/APEC_spectra_0.15-15.0keV_LightconesCE.hdf5', 'r')
    temptab   = APEC_spec['Log_Plasma_Temp'][:]
    energies  = APEC_spec['Energies'][:]
    APECtab   = {}
    APECtab['Hydrogen']  = APEC_spec['Hydrogen'][:]
    APECtab['Helium']    = APEC_spec['Helium'][:]
    APECtab['Carbon']    = APEC_spec['Carbon'][:]
    APECtab['Nitrogen']  = APEC_spec['Nitrogen'][:]
    APECtab['Oxygen']    = APEC_spec['Oxygen'][:]
    APECtab['Neon']      = APEC_spec['Neon'][:]
    APECtab['Magnesium'] = APEC_spec['Magnesium'][:]
    APECtab['Silicon']   = APEC_spec['Silicon'][:]
    APECtab['Sulphur']   = APEC_spec['Sulphur'][:]
    APECtab['Calcium']   = APEC_spec['Calcium'][:]
    APECtab['Iron']      = APEC_spec['Iron'][:]

    data['halo_'+x]['SpecEngr'] = energies
    data['halo_'+x]['SpecTemp'] = temptab
    data['halo_'+x]['SpecAPEC'] = APECtab
    
    ne_nH = np.zeros(len(data['halo_'+x]['GASpos_p']))+1
    ni_nH = np.zeros(len(data['halo_'+x]['GASpos_p']))+1
    mu    = np.zeros(len(data['halo_'+x]['GASpos_p']))

    # -- Sum element contributions
    # Hydrogen
    H       = data['halo_'+x]['H_p']
    mu     += 1.0/(1.0+1.0)
    lN_H_AG = 12.00
    # Helium
    He_H   = data['halo_'+x]['He_p']/H
    ne_nH += (He_H)*(1.00794/4.002602)*(2.0/1.0)
    ni_nH += (He_H)*(1.00794/4.002602)
    mu    += (He_H)/(1.0+2.0)
    AG_He  = 10.99-lN_H_AG
    He_H=10.0**(np.log10(He_H*(1.00794/4.002602))-AG_He)
    # Carbon
    C_H    = data['halo_'+x]['C_p']/H
    ne_nH += (C_H)*(1.00794/12.0107)*(6.0/1.0)
    ni_nH += (C_H)*(1.00794/12.0107)
    mu    += (C_H)/(1.0+6.0)
    AG_C   = 8.56-lN_H_AG
    C_H    = 10.0**(np.log10(C_H*(1.00794/12.0107))-AG_C)
    # Nitrogen
    N_H    = data['halo_'+x]['N_p']/H
    ne_nH += (N_H)*(1.00794/14.0067)*(7.0/1.0)
    ni_nH += (N_H)*(1.00794/14.0067)
    mu    += (N_H)/(1.0+7.0)
    AG_N   = 8.05-lN_H_AG
    N_H    = 10.0**(np.log10(N_H*(1.00794/14.0067))-AG_N)
    # Oxygen
    O_H    = data['halo_'+x]['O_p']/H
    ne_nH += (O_H)*(1.00794/15.9994)*(8.0/1.0)
    ni_nH += (O_H)*(1.00794/15.9994)
    mu    += (O_H)/(1.0+8.0)
    AG_O   = 8.93-lN_H_AG
    O_H    = 10.0**(np.log10(O_H*(1.00794/15.9994))-AG_O)
    # Neon
    Ne_H   = data['halo_'+x]['Ne_p']/H
    ne_nH += (Ne_H)*(1.00794/20.1797)*(10.0/1.0)
    ni_nH += (Ne_H)*(1.00794/20.1797)
    mu    += (Ne_H)/(1.0+10.0)
    AG_Ne  = 8.09-lN_H_AG
    Ne_H   = 10.0**(np.log10(Ne_H*(1.00794/20.1797))-AG_Ne)
    # Magnesium
    Mg_H   = data['halo_'+x]['Mg_p']/H
    ne_nH += (Mg_H)*(1.00794/24.3050)*(12.0/1.0)
    ni_nH += (Mg_H)*(1.00794/24.3050)
    mu    += (Mg_H)/(1.0+12.0)
    AG_Mg  = 7.58-lN_H_AG
    Mg_H   = 10.0**(np.log10(Mg_H*(1.00794/24.3050))-AG_Mg)
    # Silicon, Sulphur & Calcium
    Si_H   = data['halo_'+x]['Si_p']/H
    Ca_Si  = 0.0941736
    S_Si   = 0.6054160
    ne_nH += (Si_H)*(1.00794/28.0855)*(14.0/1.0)
    ne_nH += (Si_H*Ca_Si)*(1.00794/40.078)*(20.0/1.0)
    ne_nH += (Si_H*S_Si)*(1.00794/32.065)*(16.0/1.0)
    ni_nH += (Si_H)*(1.00794/28.0855)
    ni_nH += (Si_H*Ca_Si)*(1.00794/40.078)
    ni_nH += (Si_H*S_Si)*(1.00794/32.065)
    mu    += (Si_H)/(1.0+14.0)
    mu    += (Si_H*Ca_Si)/(1.0+20.0)
    mu    += (Si_H*S_Si)/(1.0+16.0)
    AG_Si  = 7.55-lN_H_AG
    AG_Ca  = 6.36-lN_H_AG
    AG_S   = 7.21-lN_H_AG
    Ca_H   = 10.0**(np.log10((Ca_Si*Si_H)*(1.00794/40.078))-AG_Ca)
    S_H    = 10.0**(np.log10((S_Si*Si_H)*(1.00794/32.065))-AG_S)
    Si_H   = 10.0**(np.log10(Si_H*(1.00794/28.0855))-AG_Si)
    # Iron
    Fe_H   = data['halo_'+x]['Fe_p']/H
    ne_nH += (Fe_H)*(1.00794/55.845)*(26.0/1.0)
    ni_nH += (Fe_H)*(1.00794/55.845)
    mu    += (Fe_H)/(1.0+26.0)
    AG_Fe  = 7.67-lN_H_AG
    Fe_H   = 10.0**(np.log10(Fe_H*(1.00794/55.845))-AG_Fe)

    # Emission measure & Y parameter
    EMM  = (data['halo_'+x]['GASrho_p']*((ne_nH/((ne_nH+ni_nH)*mu*bsc.mp))**2.0))*data['halo_'+x]['GASmass_p']/ne_nH

    Ypar = (bsc.sigma_t/(511.0*bsc.erg2keV))*bsc.kb*data['halo_'+x]['GAStemp_p']*(data['halo_'+x]['GASmass_p']*0.752*ne_nH/bsc.mp)/bsc.mpc/bsc.mpc # Mpc^2

    # Remove cold, dense gas
    idx = np.where((data['halo_'+x]['GAStemp_p'] <= 10.0**5.2) | (data['halo_'+x]['GASrho_p']*0.752/bsc.mp >= 0.1) | (data['halo_'+x]['GASeos_p'] > 0.999))[0]
    if len(idx) > 0:
        EMM[idx]  = 0.0
        Ypar[idx] = 0.0
    del idx

    # Calculate spectrum
    idx  = np.where((np.absolute(data['halo_'+x]['GASgrp_p'].astype(np.int))-1 == data['halo_'+x]['FOFh']) &
                    ((data['halo_'+x]['GASsgp_p'] == 0) | (data['halo_'+x]['GASsgp_p'] == np.max(data['halo_'+x]['GASsgp_p']))) &
                    (data['halo_'+x]['GAStemp_p'] >= 10.0**5.2) &
                    (data['halo_'+x]['GASrho_p']*(0.752/bsc.mp) <= 0.1))[0]

    pos  = data['halo_'+x]['GASpos_p'][idx]-data['halo_'+x]['CoP']
    r    = np.sqrt(pos[:,0]*pos[:,0]+pos[:,1]*pos[:,1]+pos[:,2]*pos[:,2])
    mass = data['halo_'+x]['GASmass_p'][idx]
    temp = data['halo_'+x]['GAStemp_p'][idx]
    iron = data['halo_'+x]['Fe_p'][idx]
    EMM  = EMM[idx]
    He_H = He_H[idx]
    C_H  = C_H[idx]
    N_H  = N_H[idx]
    O_H  = O_H[idx]
    Ne_H = Ne_H[idx]
    Mg_H = Mg_H[idx]
    Ca_H = Ca_H[idx]
    S_H  = S_H[idx]
    Si_H = Si_H[idx]
    Fe_H = Fe_H[idx]
    del idx

    nbins = 25+1
    rm    = np.log10(0.001*data['halo_'+x]['R200'])
    rx    = np.log10(5.0*data['halo_'+x]['R200'])
    rbin  = np.logspace(rm,rx,num=nbins,base=10.0)
    rcen  = 10.0**(0.5*np.log10(rbin[1:]*rbin[:-1]))
    vol   = (4.0/3.0)*np.pi*((rbin[1:]**3.0)-(rbin[:-1]**3.0))

    mpro = bsc.radial_bin(comm,NProcs,MyRank,pos,mass,rmin=np.min(rbin),rmax=np.max(rbin),nb=nbins)[1]
    data['halo_'+x]['Srho'] = mpro/vol
    data['halo_'+x]['Svol'] = vol
    tpro = bsc.radial_bin(comm, NProcs, MyRank, pos, mass*temp, rmin=np.min(rbin), rmax=np.max(rbin), nb=nbins)[1]
    data['halo_'+x]['Stmp'] = (tpro/mpro)*(bsc.kb/bsc.erg2keV)
    zpro = bsc.radial_bin(comm, NProcs, MyRank, pos, mass*iron, rmin=np.min(rbin), rmax=np.max(rbin), nb=nbins)[1]
    data['halo_'+x]['Smet'] = (zpro/mpro)/1.29e-3
    del pos, vol, mpro, tpro, zpro

    spectrum=np.zeros((len(rcen),len(energies)))

    for k in xrange(0, len(rcen), 1):
        idx = np.where((r > rbin[k]) & (r <= rbin[k+1]))[0]
        if len(idx) <= 0: continue
        itemp = bsc.locate(temptab, np.log10(temp[idx]))
        for j in xrange(0, len(energies), 1):
            spectrum[k,j] += np.sum(EMM[idx]*(APECtab['Hydrogen'][itemp,j]+APECtab['Helium'][itemp,j]*He_H[idx]+APECtab['Carbon'][itemp,j]*C_H[idx]+APECtab['Nitrogen'][itemp,j]*N_H[idx]+APECtab['Oxygen'][itemp,j]*O_H[idx]+APECtab['Neon'][itemp,j]*Ne_H[idx]+APECtab['Magnesium'][itemp,j]*Mg_H[idx]+APECtab['Silicon'][itemp,j]*Si_H[idx]+APECtab['Sulphur'][itemp,j]*S_H[idx]+APECtab['Calcium'][itemp,j]*Ca_H[idx]+APECtab['Iron'][itemp,j]*Fe_H[idx]))

    spectrum = bsc.coresum(comm, MyRank, NProcs, spectrum.reshape(-1,1))
    spectrum = spectrum.reshape(len(rcen),len(energies))

    del r, temp, He_H, C_H, N_H, O_H, Ne_H, Mg_H, Ca_H, S_H, Si_H, Fe_H, rm, rx
    del data['halo_'+x]['GASpos_p'], data['halo_'+x]['GASmass_p'], data['halo_'+x]['GASrho_p'], data['halo_'+x]['GAStemp_p'], data['halo_'+x]['GASeos_p'], data['halo_'+x]['GASgrp_p'], data['halo_'+x]['GASsgp_p'], data['halo_'+x]['H_p'], data['halo_'+x]['He_p'], data['halo_'+x]['C_p'], data['halo_'+x]['N_p'], data['halo_'+x]['O_p'], data['halo_'+x]['Ne_p'], data['halo_'+x]['Mg_p'], data['halo_'+x]['Si_p'], data['halo_'+x]['Fe_p']

    data['halo_'+x]['Rspec']    = rcen
    data['halo_'+x]['Spectrum'] = spectrum
    data['halo_'+x]['EMM']      = EMM
    data['halo_'+x]['Ypar']     = Ypar
    del rcen, spectrum, EMM, Ypar
    return

def fit_spectrum(comm, NProcs, MyRank, x, data):
    import basics as bsc
    import numpy as np

    chand_area = np.loadtxt('src/chandra_acis-i_.area')
    etmp = chand_area[:,0]
    atmp = chand_area[:,1]
    idxa = bsc.locate(etmp, data['halo_'+x]['SpecEngr'])
    aeff = atmp[idxa]
    del idxa
    DL   = 250.0*bsc.mpc
    tint = 1.0e6
    mabs = bsc.wabs(data['halo_'+x]['SpecEngr'],2.0e20)
    spec_sfac = mabs*aeff*tint/(4.0*np.pi*DL*DL)/bsc.erg2keV
    data['halo_'+x]['SpecSFac'] = spec_sfac
    del chand_area, etmp, atmp, aeff, DL, tint, mabs

    nbins = len(data['halo_'+x]['Spectrum'])
    temp  = np.zeros(nbins)
    rho   = np.zeros(nbins)
    zmet  = np.zeros(nbins)
    xisq  = np.zeros(nbins)
    for k in xrange(0, nbins, 1):
        spectrum = data['halo_'+x]['Spectrum'][k]

        if np.max(spectrum) <= 0.0:
            temp[k] = 0.0
            rho[k]  = 0.0
            zmet[k] = 0.0
            continue

        vol = data['halo_'+x]['Svol'][k]
        Tg  = np.log10(data['halo_'+x]['Stmp'][k]*(bsc.erg2keV/bsc.kb))
        Zg  = data['halo_'+x]['Smet'][k]*(1.29e-3/1.89e-3)
        Dg  = (data['halo_'+x]['Srho'][k]**2.0)*(vol/1.0e66)*(((bsc.Xe(Zg)/((bsc.Xe(Zg)+bsc.Xi(Zg))*bsc.mu2(Zg)*bsc.mp))**2.0)/bsc.Xe(Zg))

        params, fitxi, spec_mod = specfit(spectrum, Tg, Dg, Zg, data['halo_'+x]['SpecEngr'], data['halo_'+x]['SpecSFac'], data['halo_'+x]['SpecTemp'], data['halo_'+x]['SpecAPEC'])

        temp[k] = params.x[0]
        rho[k]  = params.x[1]
        zmet[k] = params.x[2]
        xisq[k] = fitxi
        del spectrum, vol, Tg, Dg, Zg, params, fitxi, spec_mod

    data['halo_'+x]['Tspec'] = (bsc.kb/bsc.erg2keV)*(10.0**temp)

    ez2      = data['halo_'+x]['OmgM']*(1.0+data['halo_'+x]['zred'])**3.0+data['halo_'+x]['OmgL']
    rho_crit = 1.878e-29*data['halo_'+x]['Hub']*data['halo_'+x]['Hub']*ez2
    rho_spec = np.zeros(len(rho))
    for k in xrange(0, len(rho), 1): rho_spec[k] = np.sqrt((rho[k]*1.0e66/data['halo_'+x]['Svol'][k])/(((bsc.Xe(zmet[k])/((bsc.Xe(zmet[k])+bsc.Xi(zmet[k]))*bsc.mu2(zmet[k])*bsc.mp))**2.0)/bsc.Xe(zmet[k])))/rho_crit
    data['halo_'+x]['RHOspec'] = rho_spec
    data['halo_'+x]['Zspec']   = zmet*(1.89e-3/1.29e-3)
    data['halo_'+x]['XIspec']  = xisq

    del nbins, rho, temp, zmet, xisq, ez2, rho_crit, rho_spec
    return

def specfit(spectrum, Tg, Dg, Zg, energies, spec_sfac, temptab, APECtab):
    import numpy as np
    from scipy.optimize import minimize

    # Calc photons per bin (1Ms)
    ppb = (spectrum/energies)*spec_sfac
    # Set initial guess & fit limits -- 6<log(T)<9, norm > 0, 0<Z/Zsun<10
    ini_guess = [Tg, Dg, Zg]
    lims      = [(6.0,9.0), (0.0,np.inf), (0.01,10.0)]
    # Fit only bins with ppb > 0 and 0.5 < E < 10.0
    tdx = np.where((ppb > 0.0) & (energies > 0.5) & (energies < 10.0))[0]
    # Initial model spectrum fit
    ebins = np.arange(len(energies))

    coef = []
    coef.append(minimize(spec_model, ini_guess, args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'), method='L-BFGS-B', bounds=lims, tol=1.0e-4, options={'maxiter':1000}))

    coef.append(minimize(spec_model, ini_guess, args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'), method='SLSQP', bounds=lims, tol=1.0e-4, options={'maxiter':500}))

    coef.append(minimize(spec_model, ini_guess, args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'), method='Nelder-Mead', tol=1.0e-4, options={'maxiter':1000}))
    if coef[-1].x[2] < 0.0: coef[-1].success = False

    try:
        coef.append(least_squares(spec_model, ini_guess, args=(ebins[tdx], ppb[tdx], temptab, APECtab, energies, spec_sfac, tdx, 'Fln'), bounds=([6.0, 0.0, 0.0], [9.0, np.inf, 10.0]), ftol=1.0e-4, max_nfev=1000))
    except:
        pass

    fitcoef = None
    xisq    = 1.0e100

    for y in coef:
        if y.success is False: continue
        spec_mod = spec_model(y.x, ppb[tdx], ebins[tdx], temptab, APECtab, energies, spec_sfac, tdx)
        Xsq_m    = np.sum((spec_mod[tdx]-ppb[tdx])**2.0/ppb[tdx]**2.0)/(len(ebins[tdx])-3)
        if Xsq_m < xisq:
            fitcoef = y
            xisq    = Xsq_m
        del Xsq_m

    if fitcoef is None:
        fitcoef   = coef[0]
        fitcoef.x = ini_guess
        spec_mod  = spec_model(fitcoef.x, ppb[tdx], ebins[tdx], temptab, APECtab, energies, spec_sfac, tdx)
        xisq      = 1.0e100
    del coef
    model_flux = spec_mod*energies/spec_sfac
    del ppb, ini_guess, lims, tdx, ebins, spec_mod
    return fitcoef, xisq, model_flux

def spec_model(p0, x, y, temptab, APECtab, energies, spec_sfac, tdx, meth='Mod'):
    import numpy as np
    import basics as bsc    

    # Locate T in look up table
    T, D, Z = p0
    idx = bsc.locate(temptab, T)
    if idx < 0: idx = 0
    if idx > len(temptab)-2: idx = len(temptab)-2
    dlogT = temptab[1]-temptab[0]
    #-- Element contributions
    # Hydrogen
    m = (np.log10(APECtab['Hydrogen'][idx+1])-np.log10(APECtab['Hydrogen'][idx]))/dlogT
    b = np.log10(APECtab['Hydrogen'][idx])-m*temptab[idx]
    H = 10.0**(m*T+b)
    # Helium
    m  = (np.log10(APECtab['Helium'][idx+1])-np.log10(APECtab['Helium'][idx]))/dlogT
    b  = np.log10(APECtab['Helium'][idx])-m*temptab[idx]
    He = 10.0**(m*T+b)
    # Carbon
    m = (np.log10(APECtab['Carbon'][idx+1])-np.log10(APECtab['Carbon'][idx]))/dlogT
    b = np.log10(APECtab['Carbon'][idx])-m*temptab[idx]
    C = 10.0**(m*T+b)
    # Nitrogen
    m = (np.log10(APECtab['Nitrogen'][idx+1])-np.log10(APECtab['Nitrogen'][idx]))/dlogT
    b = np.log10(APECtab['Nitrogen'][idx])-m*temptab[idx]
    N = 10.0**(m*T+b)
    # Oxygen
    m = (np.log10(APECtab['Oxygen'][idx+1])-np.log10(APECtab['Oxygen'][idx]))/dlogT
    b = np.log10(APECtab['Oxygen'][idx])-m*temptab[idx]
    O = 10.0**(m*T+b)
    # Neon
    m  = (np.log10(APECtab['Neon'][idx+1])-np.log10(APECtab['Neon'][idx]))/dlogT
    b  = np.log10(APECtab['Neon'][idx])-m*temptab[idx]
    Ne = 10.0**(m*T+b)
    # Magnesium
    m  = (np.log10(APECtab['Magnesium'][idx+1])-np.log10(APECtab['Magnesium'][idx]))/dlogT
    b  = np.log10(APECtab['Magnesium'][idx])-m*temptab[idx]
    Mg = 10.0**(m*T+b)
    # Silcon
    m  = (np.log10(APECtab['Silicon'][idx+1])-np.log10(APECtab['Silicon'][idx]))/dlogT
    b  = np.log10(APECtab['Silicon'][idx])-m*temptab[idx]
    Si = 10.0**(m*T+b)
    # Sulphur
    m = (np.log10(APECtab['Sulphur'][idx+1])-np.log10(APECtab['Sulphur'][idx]))/dlogT
    b = np.log10(APECtab['Sulphur'][idx])-m*temptab[idx]
    S = 10.0**(m*T+b)
    # Calcium
    m  = (np.log10(APECtab['Calcium'][idx+1])-np.log10(APECtab['Calcium'][idx]))/dlogT
    b  = np.log10(APECtab['Calcium'][idx])-m*temptab[idx]
    Ca = 10.0**(m*T+b)
    # Iron
    m  = (np.log10(APECtab['Iron'][idx+1])-np.log10(APECtab['Iron'][idx]))/dlogT
    b  = np.log10(APECtab['Iron'][idx])-m*temptab[idx]
    Fe = 10.0**(m*T+b)
    #-- Model spectrum
    mod = D*(H+He+Z*(C+N+O+Ne+Mg+Si+S+Ca+Fe))
    mod = 1.0e66*(mod/energies)*spec_sfac
    #-- Return error or model
    if meth == 'Flg':
        return np.sum((np.log10(y)-np.log10(mod[tdx]))**2.0)
    elif meth == 'Fln':
        return np.sum((y-mod[tdx])**2.0)
    else:
        return mod

def cool_func_soft(comm, NProcs, MyRank, x, data, pix):
    import read
    import basics as bsc
    import numpy as np
    from scipy.io.idl import readsav
    np.seterr(divide='ignore')

    ck = read.get_data(comm, NProcs, MyRank, x, 'CoolFunc', data, [])

    APEC = readsav('src/APEC_0.5_2.0keV_interp.idl')
    
    inde = 0 # 0 - erg/s, 1 - photons
    indz = bsc.locate(APEC['redshift'], data['halo_'+x]['zred'])
    indT = bsc.locate(APEC["ltemp"], np.log10(data['halo_'+x]['GAStemp']))

    ne_nH  = np.zeros(len(data['halo_'+x]['GASpos']))+1
    ni_nH  = np.zeros(len(data['halo_'+x]['GASpos']))+1
    mu     = np.zeros(len(data['halo_'+x]['GASpos']))
    Lambda = np.zeros(len(data['halo_'+x]['GASpos']), dtype=np.float64)

    #--- Sum element contributions
    # Hydrogen
    H       = data['halo_'+x]['H']
    mu     += 1.0/(1.0+1.0)
    lN_H_AG = 12.00
    Lambda += APEC["Lambda_hydrogen"][indz, indT, inde]
    # Helium
    He_H    = data['halo_'+x]['He']/H
    ne_nH  += (He_H)*(1.00794/4.002602)*(2.0/1.0)
    ni_nH  += (He_H)*(1.00794/4.002602)
    mu     += (He_H)/(1.0+2.0)
    AG_He   = 10.99-lN_H_AG
    He_H    = 10.0**(np.log10(He_H*(1.00794/4.002602))-AG_He)
    Lambda += He_H*APEC["Lambda_helium"][indz, indT, inde]
    del He_H
    # Carbon
    C_H     = data['halo_'+x]['C']/H
    ne_nH  += (C_H)*(1.00794/12.0107)*(6.0/1.0)
    ni_nH  += (C_H)*(1.00794/12.0107)
    mu     += (C_H)/(1.0+6.0)
    AG_C    = 8.56-lN_H_AG
    C_H     = 10.0**(np.log10(C_H*(1.00794/12.0107))-AG_C)
    C_H     = 10.0**(np.log10(C_H*(1.00794/12.0107))-AG_C)
    Lambda += C_H*APEC["Lambda_carbon"][indz, indT, inde]
    # Nitrogen
    N_H     = data['halo_'+x]['N']/H
    ne_nH  += (N_H)*(1.00794/14.0067)*(7.0/1.0)
    ni_nH  += (N_H)*(1.00794/14.0067)
    mu     += (N_H)/(1.0+7.0)
    AG_N    = 8.05-lN_H_AG
    N_H     = 10.0**(np.log10(N_H*(1.00794/14.0067))-AG_N)
    Lambda += N_H*APEC["Lambda_nitrogen"][indz, indT, inde]
    del N_H
    # Oxygen
    O_H     = data['halo_'+x]['O']/H
    ne_nH  += (O_H)*(1.00794/15.9994)*(8.0/1.0)
    ni_nH  += (O_H)*(1.00794/15.9994)
    mu     += (O_H)/(1.0+8.0)
    AG_O    = 8.83-lN_H_AG
    O_H     = 10.0**(np.log10(O_H*(1.00794/15.9994))-AG_O)
    Lambda += O_H*APEC["Lambda_oxygen"][indz, indT, inde]
    del O_H
    # Neon
    Ne_H    = data['halo_'+x]['Ne']/H
    ne_nH  += (Ne_H)*(1.00794/20.1797)*(10.0/1.0)
    ni_nH  += (Ne_H)*(1.00794/20.1797)
    mu     += (Ne_H)/(1.0+10.0)
    AG_Ne   = 8.09-lN_H_AG
    Ne_H    = 10.0**(np.log10(Ne_H*(1.00794/20.1797))-AG_Ne)
    Lambda += Ne_H*APEC["Lambda_neon"][indz, indT, inde]
    del Ne_H
    # Magnesium
    Mg_H    = data['halo_'+x]['Mg']/H
    ne_nH  += (Mg_H)*(1.00794/24.3050)*(12.0/1.0)
    ni_nH  += (Mg_H)*(1.00794/24.3050)
    mu     += (Mg_H)/(1.0+12.0)
    AG_Mg   = 7.58-lN_H_AG
    Mg_H    = 10.0**(np.log10(Mg_H*(1.00794/24.3050))-AG_Mg)
    Lambda += Mg_H*APEC["Lambda_magnesium"][indz, indT, inde]
    del Mg_H
    # Silicon, Sulphur & Calcium
    Si_H    = data['halo_'+x]['Si']/H
    Ca_Si   = 0.0941736
    S_Si    = 0.6054160
    ne_nH  += (Si_H)*(1.00794/28.0855)*(14.0/1.0)
    ne_nH  += (Si_H*Ca_Si)*(1.00794/40.078)*(20.0/1.0)
    ne_nH  += (Si_H*S_Si)*(1.00794/32.065)*(16.0/1.0)
    ni_nH  += (Si_H)*(1.00794/28.0855)
    ni_nH  += (Si_H*Ca_Si)*(1.00794/40.078)
    ni_nH  += (Si_H*S_Si)*(1.00794/32.065)
    mu     += (Si_H)/(1.0+14.0)
    mu     += (Si_H*Ca_Si)/(1.0+20.0)
    mu     += (Si_H*S_Si)/(1.0+16.0)
    AG_Si   = 7.55-lN_H_AG
    AG_Ca   = 6.36-lN_H_AG
    AG_S    = 7.21-lN_H_AG
    Ca_H    = 10.0**(np.log10((Ca_Si*Si_H)*(1.00794/40.078))-AG_Ca)
    S_H     = 10.0**(np.log10((S_Si*Si_H)*(1.00794/32.065))-AG_S)
    Si_H    = 10.0**(np.log10(Si_H*(1.00794/28.0855))-AG_Si)
    Lambda += Si_H*APEC["Lambda_silicon"][indz, indT, inde]
    Lambda += Ca_H*APEC["Lambda_calcium"][indz, indT, inde]
    Lambda += S_H*APEC["Lambda_sulphur"][indz, indT, inde]
    del Si_H, Ca_H, S_H
    # Iron
    Fe_H    = data['halo_'+x]['Fe']/H
    ne_nH  += (Fe_H)*(1.00794/55.845)*(26.0/1.0)
    ni_nH  += (Fe_H)*(1.00794/55.845)
    mu     += (Fe_H)/(1.0+26.0)
    AG_Fe   = 7.67-lN_H_AG
    Fe_H    = 10.0**(np.log10(Fe_H*(1.00794/55.845))-AG_Fe)
    Lambda += Fe_H*APEC["Lambda_iron"][indz, indT, inde]
    del H, Fe_H, indT

    #--- Calculate observables
    Lx   = Lambda*(data['halo_'+x]['GASrho']*(ne_nH/((ne_nH+ni_nH)*mu*bsc.mp))**2.0)*data['halo_'+x]['GASmass']/ne_nH
    Sx   = Lx/(4.0*np.pi*pix*pix)/((180.0*60.0/np.pi)**2)
    Ypix = (bsc.sigma_t/(511.0*bsc.erg2keV))*bsc.kb*data['halo_'+x]['GAStemp']*(data['halo_'+x]['GASmass']/(mu*bsc.mp))*(ne_nH/(ne_nH+ni_nH))/(pix*pix)
    del Lx

    # Remove cold, dense gas
    idx = np.where((data['halo_'+x]['GAStemp'] <= 10.0**5.2) | (data['halo_'+x]['GASrho']*0.752/bsc.mp >= 0.1) | (data['halo_'+x]['GASeos'] > 0.999))[0]
    if len(idx) > 0:
        Sx[idx]   = 0.0
        Ypix[idx] = 0.0
    del idx

    data['halo_'+x]['Sx']   = Sx
    data['halo_'+x]['Ypix'] = Ypix
    del Sx, Ypix
    return
