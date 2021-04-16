import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

class MetallicityScale:
    mass_h = 1.00794
    mass_fe = 55.845
    mass_o = 15.999
    mass_si = 28.0855

    def AG89(self):
        X_sol = 0.70643
        Y_sol = 0.27416
        Z_sol = 0.01941

        # log(Fe)+12 = 7.67
        Nfe_h = 10.0 ** (7.67 - 12.0)
        Zfe_h = Nfe_h * (self.mass_fe / self.mass_h)
        Zfe_tot = Zfe_h * X_sol

        # log(O)+12 = 8.93
        No_h = 10.0 ** (8.93 - 12.0)
        Zo_h = No_h * (self.mass_o / self.mass_h)
        Zo_tot = Zo_h * X_sol

        # log(Si)+12 = 7.55
        Nsi_h = 10.0 ** (7.55 - 12.0)
        Zsi_h = Nsi_h * (self.mass_si / self.mass_h)
        Zsi_tot = Zsi_h * X_sol

        return Z_sol, np.around(Zfe_tot, decimals=5), np.around(Zo_tot, decimals=5), np.around(Zsi_tot, decimals=6)

    def Asplund09(self):
        X_sol = 0.7381
        Y_sol = 0.2485
        Z_sol = 0.0134

        # log(Fe)+12 = 7.50
        Nfe_h = 10.0 ** (7.50 - 12.0)
        Zfe_h = Nfe_h * (self.mass_fe / self.mass_h)
        Zfe_tot = Zfe_h * X_sol

        # log(O)+12 = 8.69
        No_h = 10.0 ** (8.69 - 12.0)
        Zo_h = No_h * (self.mass_o / self.mass_h)
        Zo_tot = Zo_h * X_sol

        # log(Si)+12 = 7.51
        Nsi_h = 10.0 ** (7.51 - 12.0)
        Zsi_h = Nsi_h * (self.mass_si / self.mass_h)
        Zsi_tot = Zsi_h * X_sol

        return Z_sol, np.around(Zfe_tot, decimals=5), np.around(Zo_tot, decimals=5), np.around(Zsi_tot, decimals=6)

    def Lodders09(self):
        X_sol = 0.7112
        Y_sol = 0.2735
        Z_sol = 0.0153

        # log(Fe)+12 = 7.46
        Nfe_h = 10.0 ** (7.46 - 12.0)
        Zfe_h = Nfe_h * (self.mass_fe / self.mass_h)
        Zfe_tot = Zfe_h * X_sol

        # log(O)+12 = 8.73
        No_h = 10.0 ** (8.73 - 12.0)
        Zo_h = No_h * (self.mass_o / self.mass_h)
        Zo_tot = Zo_h * X_sol

        # log(Si)+12 = 7.53
        Nsi_h = 10.0 ** (7.53 - 12.0)
        Zsi_h = Nsi_h * (self.mass_si / self.mass_h)
        Zsi_tot = Zsi_h * X_sol

        return Z_sol, np.around(Zfe_tot, decimals=5), np.around(Zo_tot, decimals=5), np.around(Zsi_tot, decimals=6)

    def GS98(self):
        X_sol = 0.735
        Y_sol = 0.248
        Z_sol = 0.017

        # log(Fe)+12 = 7.50
        Nfe_h = 10.0 ** (7.50 - 12.0)
        Zfe_h = Nfe_h * (self.mass_fe / self.mass_h)
        Zfe_tot = Zfe_h * X_sol

        # log(O)+12 = 8.83
        No_h = 10.0 ** (8.83 - 12.0)
        Zo_h = No_h * (self.mass_o / self.mass_h)
        Zo_tot = Zo_h * X_sol

        # log(Si)+12 = 7.55
        Nsi_h = 10.0 ** (7.55 - 12.0)
        Zsi_h = Nsi_h * (self.mass_si / self.mass_h)
        Zsi_tot = Zsi_h * X_sol

        return Z_sol, np.around(Zfe_tot, decimals=5), np.around(Zo_tot, decimals=5), np.around(Zsi_tot, decimals=6)


class Yates17(MetallicityScale):
    kT500 = [np.average([2.01, 2.26, 1.96, 2.00, 2.14]),
             np.average([4.21, 4.56, 4.17, 3.87]),
             np.average([3.73, 3.78]),
             2.63, 0.91,
             np.average([1.44, 1.29, 1.47, 1.67]),
             4.14, 1.54,
             np.average([4.53, 4.46, 4.34, 4.07]),
             4.61,
             np.average([2.76, 2.95, 2.88, 2.94, 2.94]),
             2.16,
             np.average([6.29, 5.34, 5.74]),
             np.average([2.16, 2.00]),
             np.average([2.37, 2.20]),
             3.07, 4.41,
             np.average([5.60, 6.14, 5.01, 5.21]),
             7.96, 2.19,
             np.average([3.93, 3.37, 3.87, 4.32, 3.87]),
             np.average([5.82, 4.81]),
             np.average([5.19, 5.94, 4.94]), np.average([2.08, 2.07, 2.28, 2.07]),
             np.average([2.46, 2.60]), 5.78, 3.28, np.average([2.74, 3.08, 3.02, 2.80]), 6.19,
             np.average([4.73, 4.65, 4.21]), np.average([5.94, 6.56]), np.average([2.14, 2.40]),
             np.average([2.47, 1.38]), np.average([3.00, 3.53]), np.average([5.99, 5.81, 5.61]), 2.66,
             np.average([2.46, 2.67]), 2.21, np.average([3.42, 4.96, 3.61]), 2.02, np.average([3.22, 3.21]),
             np.average([4.49, 4.34]), 1.31, np.average([4.19, 3.67]), 6.00, 2.00,
             np.average([2.65, 3.68, 2.67, 2.65]), np.average([2.50, 2.40]), 0.80, 0.54,
             np.average([0.70, 0.67, 0.67]), 0.60, np.average([2.38, 3.68, 2.27, 2.34]), 0.44,
             np.average([2.46, 2.27, 2.34, 2.66, 2.47]), np.average([1.15, 1.19, 1.19]), 1.11,
             np.average([0.84, 0.87]), np.average([0.92, 0.87, 0.82]), np.average([0.91, 0.95]), 0.68, 0.72,
             np.average([0.59, 0.56, 0.52]), 0.44, 0.22, 0.83, 0.66, 0.57, np.average([0.72, 0.90, 0.75, 0.75]),
             0.70, 0.64, np.average([0.51, 0.44]), 1.42, 0.71, 0.84, 6.85, 5.56, np.average([2.33, 1.60, 1.89]),
             6.71]

    Fe = [np.average([0.31, 0.25, 0.29, 0.43, 0.43]), np.average([0.26, 0.25, 0.35, 0.41]),
          np.average([0.26, 0.25]), 0.48, 0.18, np.average([0.30, 0.33, 0.37, 0.39]), 0.31, 0.36,
          np.average([0.35, 0.26, 0.37, 0.33]), 0.19, np.average([0.30, 0.26, 0.45, 0.36, 0.35]), 0.20,
          np.average([0.27, 0.34, 0.27]), np.average([0.30, 0.28]), np.average([0.21, 0.45]), 0.37, 0.29,
          np.average([0.25, 0.24, 0.27, 0.23]), 0.26, 0.48, np.average([0.23, 0.23, 0.29, 0.31, 0.36]),
          np.average([0.16, 0.35]), np.average([0.28, 0.35, 0.43]), np.average([0.35, 0.42, 0.40, 0.42]),
          np.average([0.23, 0.53]), 0.22, 0.34, np.average([0.30, 0.24, 0.35, 0.31]), 0.35,
          np.average([0.25, 0.27, 0.32]), np.average([0.17, 0.32]), np.average([0.43, 0.53]),
          np.average([0.24, 0.24]), np.average([0.34, 0.42]), np.average([0.23, 0.28, 0.34]), 0.24,
          np.average([0.44, 0.37]), 0.21, np.average([0.20, 0.31, 0.32]), 0.29, np.average([0.26, 0.45]),
          np.average([0.23, 0.34]), 0.39, np.average([0.28, 0.28]), 0.24, 0.33,
          np.average([0.38, 0.33, 0.43, 0.43]), np.average([0.33, 0.45]), 0.22, 0.15,
          np.average([0.17, 0.13, 0.15]), 0.16, np.average([0.22, 0.27, 0.29, 0.28]), 0.07,
          np.average([0.27, 0.31, 0.31, 0.34, 0.45]), np.average([0.31, 0.24, 0.51]), 0.26,
          np.average([0.34, 0.28]), np.average([0.19, 0.27, 0.17]), np.average([0.51, 0.10]), 0.18, 0.10,
          np.average([0.17, 0.33, 0.18]), 0.15, 0.15, 0.12, 0.18, 0.10, np.average([0.30, 0.27, 0.21, 0.12]), 0.15,
          0.36, np.average([0.14, 0.18]), 0.18, 0.16, 0.10, 0.19, 0.24, np.average([0.29, 0.23, 0.33]), 0.19]

    def __init__(self, *args, **kwargs):
        super(Yates17, self).__init__(*args, **kwargs)
        Z_sol, Fe_sol, O_sol, Si_sol = self.GS98()
        self.kT500 = np.asarray(self.kT500)
        self.Fe = np.asarray(self.Fe) * Fe_sol / Fe_sol


class Mernier17(MetallicityScale):
    '''
    All data taken from CHEERS catalogue.
    44 nearby cool-core galaxy clusters, groups and ellipticals.
    23 clusters: mean temperature r<0.05 r500 greater than 1.7 keV .
    21 groups: mean temperature r<0.05 r500 lower than 1.7 keV.
    Scaled by Lodders+09.
    '''
    r_tot_500 = np.asarray(
        [3.75e-3, 0.01075, 0.017, 0.025, 0.035, 0.04525, 0.06, 0.0775, 0.1, 0.1275, 0.1475, 0.18, 0.215, 0.26, 0.425,
         0.885])
    # r_tot_500 = np.asarray([0.0075, 0.014, 0.02, 0.03, 0.04, 0.055, 0.065, 0.09, 0.11, 0.135, 0.16, 0.2, 0.23, 0.3, 0.55, 1.22])
    r_cl_500 = np.asarray([0.009, 0.029, 0.054, 0.084, 0.14, 0.21, 0.29, 0.42, 0.86])
    r_gr_500 = np.asarray([0.0045, 0.0165, 0.033, 0.053, 0.082, 0.125, 0.205, 0.615])

    Fe_tot = np.asarray(
        [0.802, 0.826, 0.825, 0.813, 0.788, 0.736, 0.684, 0.627, 0.568, 0.520, 0.480, 0.440, 0.421, 0.380, 0.304,
         0.205])
    Fe_tot_err = np.asarray(
        [0.261, 0.219, 0.197, 0.177, 0.160, 0.149, 0.129, 0.124, 0.099, 0.104, 0.104, 0.096, 0.082, 0.086, 0.090,
         0.105])
    Fe_cl = np.asarray([0.822, 0.8167, 0.7190, 0.626, 0.511, 0.432, 0.357, 0.309, 0.211])
    Fe_cl_err = np.asarray([0.241, 0.1725, 0.1369, 0.106, 0.089, 0.075, 0.081, 0.079, 0.102])
    Fe_gr = np.asarray([0.812, 0.779, 0.685, 0.640, 0.524, 0.430, 0.330, 0.268])
    Fe_gr_err = np.asarray([0.199, 0.130, 0.189, 0.175, 0.175, 0.129, 0.133, 0.139])

    O_tot = np.asarray(
        [0.437, 0.624, 0.650, 0.685, 0.632, 0.533, 0.54, 0.480, 0.42, 0.38, 0.38, 0.38, 0.33, 0.26, 0.27, 0.01])
    O_cl = np.asarray([0.815, 0.776, 0.689, 0.59, 0.46, 0.35, 0.34, 0.37, -0.02])
    O_gr = np.asarray([0.384, 0.613, 0.591, 0.460, 0.366, 0.309, 0.327, 0.19])

    Si_tot = np.asarray(
        [0.76, 0.79, 0.78, 0.77, 0.69, 0.63, 0.58, 0.53, 0.47, 0.43, 0.41, 0.371, 0.36, 0.31, 0.26, 0.10])
    Si_cl = np.asarray([0.79, 0.75, 0.61, 0.53, 0.43, 0.37, 0.31, 0.27, 0.13])
    Si_gr = np.asarray([0.76, 0.80, 0.67, 0.60, 0.49, 0.40, 0.34, 0.17])

    def __init__(self, *args, **kwargs):
        super(Mernier17, self).__init__(*args, **kwargs)
        Z_sol, Fe_sol, O_sol, Si_sol = self.Lodders09()

        hubble_parameter = Article().cosmo_model.h

        self.r_500 = [
            self.r_tot_500 * (hubble_parameter / 0.7),
            self.r_cl_500 * (hubble_parameter / 0.7),
            self.r_gr_500 * (hubble_parameter / 0.7)
        ]
        self.Fe = [
            self.Fe_tot * Fe_sol / Fe_sol,
            self.Fe_cl * Fe_sol / Fe_sol,
            self.Fe_gr * Fe_sol / Fe_sol
        ]
        self.Fe_err = [
            self.Fe_tot_err * Fe_sol / Fe_sol,
            self.Fe_cl_err * Fe_sol / Fe_sol,
            self.Fe_gr_err * Fe_sol / Fe_sol
        ]
        self.O = [
            self.O_tot * O_sol / O_sol,
            self.O_cl * O_sol / O_sol,
            self.O_gr * O_sol / O_sol
        ]
        self.Si = [
            self.Si_tot * Si_sol / Si_sol,
            self.Si_cl * Si_sol / Si_sol,
            self.Si_gr * Si_sol / Si_sol
        ]