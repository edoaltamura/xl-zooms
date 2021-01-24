import numpy as np
from scipy.optimize import minimize, least_squares, curve_fit
import unyt

# Physical constants
Gnew = 6.674e-8  # Newton's G [cm^3/gs^2]
mp = 1.6726e-24  # Proton mass [g]
kb = 1.3807e-16  # Boltzmann's constant [erg/K]
msun = 1.989e33  # Solar mass [g]
mpc = 3.0857e24  # Megaparsec [cm]
sigma_t = 6.6524e-25  # Thomson cross-section [cm^2]
erg2keV = 1.60215e-9  # erg --> keV conversion factor
mu = 0.59  # Mean molecular weight


def HSE_fit(rcen, rho, temp, x, data):

    cfr = density_fit(rcen, np.log10(rho), data['halo_' + x]['R500'])
    cft = temperature_fit(rcen, temp, data['halo_' + x]['M500'], data['halo_' + x]['R500'])

    Thse = Tmod(rcen, cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7])
    dThse = equation_of_state_dlogkT_dlogr(rcen, cft.x[0], cft.x[1], cft.x[2], cft.x[3], cft.x[4], cft.x[5], cft.x[6], cft.x[7])
    dRHOhse = equation_of_state_dlogrho_dlogr(rcen, cfr.x[0], cfr.x[1], cfr.x[2], cfr.x[3], cfr.x[4], cfr.x[5])
    Mhse = -3.68e13 * rcen * Thse * (dRHOhse + dThse)
    del Thse, dRHOhse, dThse
    return cfr, cft, Mhse


def density_fit(x, y, R500):

    p0 = [100.0, 0.1, 1.0, 1.0, 0.8 * R500 / mpc, 1.0]
    coeff_rho = minimize(
        resid_rho, p0, args=(y, x), method='L-BFGS-B',
        bounds=[
            (1.0e2, 1.0e4), (0.0, 10.0), (0.0, 10.0), (0.0, np.inf), (0.2 * R500 / mpc, np.inf), (0.0, 5.0)
        ],
        options={'maxiter': 200, 'ftol': 1e-10}
    )
    return coeff_rho


def temperature_fit(x, y, M500, R500):

    kT500 = (6.67259e-8 * M500 * mu * mp) / (2.0 * (R500)) / erg2keV

    p0 = [kT500, R500 / mpc, 0.1, 3.0, 1.0, 0.1, 1.0, kT500]
    bnds = ([0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0e-10, 0.0],
            [np.inf, np.inf, 3.0, 5.0, 10.0, np.inf, 3.0, np.inf])

    cf1 = least_squares(resid_vkt, p0, bounds=bnds, args=(y, x), max_nfev=2000)
    mod1 = Tmod(x, cf1.x[0], cf1.x[1], cf1.x[2], cf1.x[3], cf1.x[4], cf1.x[5], cf1.x[6], cf1.x[7])
    xis1 = np.sum((mod1 - y) ** 2.0 / y) / len(y)

    cf2 = minimize(resid_vkt, p0, args=(y, x), method='Nelder-Mead', options={'maxiter': 2000, 'ftol': 1e-5})
    mod2 = Tmod(x, cf2.x[0], cf2.x[1], cf2.x[2], cf2.x[3], cf2.x[4], cf2.x[5], cf2.x[6], cf2.x[7])
    xis2 = np.sum((mod2 - y) ** 2.0 / y) / len(y)

    # Assume that Nelder-Mead method works, but check xisq against bounded fit
    coeff_temp = cf2
    if xis1 < xis2:
        coeff_temp = cf1

    return coeff_temp


def resid_vkt(free_parameters, y, x):

    T0, rt, a, b, c, rcool, acool, Tmin = free_parameters
    t = T0 * (x / rt) ** (-a) / ((1 + (x / rt) ** b) ** (c / b))
    x1 = (x / rcool) ** acool
    tcool = (x1 + Tmin / T0) / (x1 + 1)
    err = (t * tcool) - y
    return np.sum(err * err)

def resid_rho(free_parameters, y, x):
    rho0, rc, alpha, beta, rs, epsilon = free_parameters
    err = np.log10(rho0 * ((x / rc) ** (-alpha * 0.5) / (1.0 + (x / rc) ** 2) ** (3 * beta / 2 - alpha / 4)) * (
            1 / ((1 + (x / rs) ** 3) ** (epsilon / 6)))) - y
    return np.sum(err * err)


def RHOmod(x, rho0, rc, alpha, beta, rs, epsilon):
    return np.log10(
        rho0 * ((x / rc) ** (-alpha / 2.0) / (1.0 + (x / rc) ** 2.0) ** (3.0 * beta / 2.0 - alpha / 4.0)) * (
                1.0 / ((1.0 + (x / rs) ** 3.0) ** (epsilon / 6.0))))

def Tmod(x, T0, rt, a, b, c, rcool, acool, Tmin):
    t = T0 * (x / rt) ** (-a) / ((1.0 + (x / rt) ** b) ** (c / b))
    x1 = (x / rcool) ** acool
    tcool = (x1 + Tmin / T0) / (x1 + 1.0)
    return t * tcool


def equation_of_state_dlogrho_dlogr(x, rho0, rc, alpha, beta, rs, epsilon):
    return -0.5 * (alpha + (6.0 * beta - alpha) * (x / rc) ** 2.0 / (1.0 + (x / rc) ** 2.0) + epsilon * \
                   (x / rs) ** 3.0 / (1.0 + (x / rs) ** 3.0))


def equation_of_state_dlogkT_dlogr(x, T0, rt, a, b, c, rcool, acool, Tmin):
    return -a + (acool * (x / rcool) ** acool / ((1.0 + (x / rcool) ** acool) * (Tmin / T0 + (x / rcool) ** acool))) * \
           (1.0 - Tmin / T0) - c * (x / rt) ** b / (1.0 + (x / rt) ** b)
