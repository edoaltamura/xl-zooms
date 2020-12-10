import numpy as np
import re
from astropy import cosmology

"""
Each cosmology has the following parameters defined:
==========  =====================================
Oc0         Omega cold dark matter at z=0
Ob0         Omega baryon at z=0
Om0         Omega matter at z=0
flat        Is this assumed flat?  If not, Ode0 must be specified
Ode0        Omega dark energy at z=0 if flat is False
H0          Hubble parameter at z=0 in km/s/Mpc
n           Density perturbation spectral index
Tcmb0       Current temperature of the CMB
Neff        Effective number of neutrino species
m_nu        Assumed mass of neutrino species, in eV.
sigma8      Density perturbation amplitude
tau         Ionisation optical depth
z_reion     Redshift of hydrogen reionisation
t0          Age of the universe in Gyr
reference   Reference for the parameters
==========  =====================================
"""

import pandas as pd
import unyt
from unyt import (
    Solar_Mass,
    Mpc,
    Dimensionless,
    Solar_Luminosity,
    Solar_Metallicity,
    keV, erg, second, K
)

# Check back-compatibility with old versions of Astropy
try:
    from astropy.cosmology import Planck18

except ImportError:

    from astropy.cosmology.core import FlatwCDM
    from astropy.cosmology.core import FlatLambdaCDM
    from astropy import units as u


    def create_cosmology(parameters=None, name=None):
        """
        A wrapper to create custom astropy cosmologies.
        The only avaliable cosmology types in this method are: FlatLambdaCDM,
        FlatwCDM, LambdaCDM and wCDM. See `astropy.cosmology`_ for more details on
        these types of cosmologies. To create a cosmology of a type that isn't
        listed above, it will have to be created directly using astropy.cosmology.
        """

        # Set the default parameters:
        params = {'H0': 70, 'Om0': 0.3, 'Oc0': 0.26, 'Ob0': 0.04, 'w0': -1,
                  'Neff': 3.04, 'flat': True, 'Tcmb0': 0.0, 'm_nu': 0.0}

        # Override default parameters with supplied parameters
        if parameters is not None:
            params.update(parameters)

        if params["flat"]:
            if params['w0'] is not -1:
                cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'],
                                 w0=params['w0'], Tcmb0=params['Tcmb0'],
                                 Neff=params['Neff'], Ob0=params['Ob0'],
                                 m_nu=u.Quantity(params['m_nu'], u.eV), name=name)

            else:
                cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'],
                                      Tcmb0=params['Tcmb0'], Neff=params['Neff'],
                                      Ob0=params['Ob0'], name=name,
                                      m_nu=u.Quantity(params['m_nu'], u.eV))
        return cosmo


    # Planck 2018 paper VI
    # Unlike Planck 2015, the paper includes massive neutrinos in Om0, which here
    # are included in m_nu.  Hence, the Om0 value differs slightly from the paper.
    planck18 = dict(
        Oc0=0.2607,
        Ob0=0.04897,
        Om0=0.3111,
        H0=67.66,
        n=0.9665,
        sigma8=0.8102,
        tau=0.0561,
        z_reion=7.82,
        t0=13.787,
        Tcmb0=2.7255,
        Neff=3.046,
        flat=True,
        m_nu=[0., 0., 0.06],
        reference=("Planck 2018 results. VI. Cosmological Parameters, A&A, submitted,"
                   " Table 2 (TT, TE, EE + lowE + lensing + BAO)")
    )


    def Planck18():
        """
        Planck18 instance of FlatLambdaCDM cosmology
        (from Planck 2018 results. VI. Cosmological Parameters,
        A&A, submitted, Table 2 (TT, TE, EE + lowE + lensing + BAO))
        """
        cosmo = create_cosmology(name="Planck18", parameters=planck18)
        return cosmo


    setattr(cosmology, "Planck18", Planck18())

unyt.define_unit("hubble_parameter", value=1. * Dimensionless, tex_repr="h")


class Observations:

    def __init__(self, cosmo_model: str = "Planck18", verbose: int = 1):

        self.verbose = verbose

        for model_name in dir(cosmology):
            if cosmo_model.lower() == model_name.lower():
                if self.verbose > 0:
                    print(f"Using the {model_name} cosmology")
                self.cosmo_model = getattr(cosmology, model_name)

    def __del__(self):
        """
        When the instance of Observations is destroyed, print out
        the name of the paper. If verbosity is 0, it has no effect.
        """
        if self.verbose > 0:
            try:
                print(f"[Literature data] Applied: {self.paper_name}")
            except:
                pass


class Sun09(Observations):
    paper_name = "Sun et al. (2009)"
    notes = "Data in h_73 units."

    M500_Sun = np.array(
        [3.18, 4.85, 3.90, 1.48, 4.85, 5.28, 8.49, 10.3,
         2.0, 7.9, 5.6, 12.9, 8.0, 14.1, 3.22, 14.9, 13.4,
         6.9, 8.95, 8.8, 8.3, 9.7, 7.9]
    ) * 1.e13 * Solar_Mass
    f500_Sun = np.array(
        [0.097, 0.086, 0.068, 0.049, 0.069, 0.060, 0.076,
         0.081, 0.108, 0.086, 0.056, 0.076, 0.075, 0.114,
         0.074, 0.088, 0.094, 0.094, 0.078, 0.099, 0.065,
         0.090, 0.093]
    )
    Mgas500_Sun = M500_Sun * f500_Sun

    def __init__(self, *args, **kwargs):
        super(Sun09, self).__init__(*args, **kwargs)

        h70_Sun = 0.73 / self.cosmo_model.h
        self.M500 = self.M500_Sun * h70_Sun
        self.Mgas500 = self.Mgas500_Sun * (h70_Sun ** 2.5)


class Lovisari15(Observations):
    paper_name = "Lovisari et al. (2015)"
    notes = "Data in h_70 units."

    M500_Lov = np.array(
        [2.07, 4.67, 2.39, 2.22, 2.95, 2.83, 3.31, 3.53, 3.49,
         3.35, 14.4, 2.34, 4.78, 8.59, 9.51, 6.96, 10.8, 4.37,
         8.00, 12.1]
    ) * 1.e13 * Solar_Mass
    Mgas500_Lov = np.array(
        [0.169, 0.353, 0.201, 0.171, 0.135, 0.272, 0.171,
         0.271, 0.306, 0.247, 1.15, 0.169, 0.379, 0.634,
         0.906, 0.534, 0.650, 0.194, 0.627, 0.817]
    ) * 1.e13 * Solar_Mass

    def __init__(self, *args, **kwargs):
        super(Lovisari15, self).__init__(*args, **kwargs)

        h70_Lov = 0.70 / self.cosmo_model.h
        self.M500 = self.M500_Lov * h70_Lov
        self.Mgas500 = self.Mgas500_Lov * (h70_Lov ** 2.5)


class Kravtsov18(Observations):
    paper_name = "Kravtsov et al. (2018)"
    notes = "Data in h_70 units."

    M500_Kra = np.array(
        [15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]
    ) * 10. * 1.e13 * Solar_Mass
    Mstar500_Kra = np.array(
        [15.34, 12.35, 8.34, 5.48, 2.68, 3.48, 2.86, 1.88, 1.85]
    ) * 0.1 * 1.e13 * Solar_Mass

    def __init__(self, *args, **kwargs):
        super(Kravtsov18, self).__init__(*args, **kwargs)

        h70_Kra = 0.70 / self.cosmo_model.h
        self.M500 = self.M500_Kra * h70_Kra
        self.Mstar500 = self.Mstar500_Kra * (h70_Kra ** 2.5)


class Budzynski14(Observations):
    paper_name = "Budzynski et al. (2014)"
    notes = ("Data in h_70 units."
             "Mstar−M500 relation",
             "a = 0.89 ± 0.14 and b = 12.44 ± 0.03",
             r"$\log \left(\frac{M_{\text {star }}}{\mathrm{M}_{\odot}}\right)=a \log \left(\frac{M_{500}}",
             r"{3 \times 10^{14} \mathrm{M}_{\odot}}\right)+b$",
             "star−M500 relation",
             "alpha = −0.11 ± 0.14 and beta = −2.04 ± 0.03",
             r"$\log f_{\text {star }}=\alpha \log \left(\frac{M_{500}}{3 \times 10^{14} "
             r"\mathrm{M}_{\odot}}\right)+\beta$")

    M500_Bud = np.array([10 ** 13.7, 10 ** 15]) * Solar_Mass
    Mstar500_Bud = 10. ** (0.89 * np.log10(M500_Bud / 3.e14) + 12.44) * Solar_Mass
    Mstar500_Bud_a = (0.89, 0.14)
    Mstar500_Bud_b = (12.44, 0.03)

    def __init__(self, *args, **kwargs):
        super(Budzynski14, self).__init__(*args, **kwargs)

        h70_Bud = 0.71 / self.cosmo_model.h
        self.M500 = self.M500_Bud * h70_Bud
        self.Mstar500 = self.Mstar500_Bud * (h70_Bud ** 2.5)
        self.fit_line_uncertainty_weights()
        self.M500_trials *= h70_Bud * Solar_Mass
        self.Mstar_trials_upper *= (h70_Bud ** 2.5) * Solar_Mass
        self.Mstar_trials_median *= (h70_Bud ** 2.5) * Solar_Mass
        self.Mstar_trials_lower *= (h70_Bud ** 2.5) * Solar_Mass

    def fit_line_uncertainty_weights(self):
        np.random.seed(0)

        # Generate random samples for the Mstar relation
        M500_trials = np.logspace(13.7, 15., 20)
        a_trials = np.random.normal(self.Mstar500_Bud_a[0], self.Mstar500_Bud_a[1], 20)
        b_trials = np.random.normal(self.Mstar500_Bud_b[0], self.Mstar500_Bud_b[1], 20)
        Mstar_trials = np.empty(0)
        for a in a_trials:
            for b in b_trials:
                Mstar_trial = 10 ** (a * np.log10(M500_trials / 3.e14) + b)
                Mstar_trials = np.append(Mstar_trials, Mstar_trial)
        Mstar_trials = Mstar_trials.reshape(-1, M500_trials.size)
        self.M500_trials = M500_trials
        self.Mstar_trials_upper = np.percentile(Mstar_trials, 16, axis=0)
        self.Mstar_trials_median = np.percentile(Mstar_trials, 50, axis=0)
        self.Mstar_trials_lower = np.percentile(Mstar_trials, 84, axis=0)


class Gonzalez13(Observations):
    paper_name = "Gonzalez et al. (2013)"
    notes = ("Data assumes WMAP7 as fiducial model. "
             "The quoted stellar baryon fractions include a deprojection correction. "
             "A0478, A2029, and A2390 are not part of the main sample. "
             "These clusters were included only in the X-ray analysis to extend the baseline to higher mass, but "
             "they have no photometry equivalent to the other systems with which to measure the stellar mass. "
             "At the high mass end, however, the stellar component contributes "
             "a relatively small fraction of the total baryons. "
             "The luminosities include appropriate e + k corrections for each galaxy from GZZ07. "
             "The stellar masses are quoted as observed, with no deprojection correction applied")

    table_masses = [
        ("ClusterID redshift TX2 Delta_TX2 L(BCG+ICL) Delta_L(BCG+ICL) LTotal Delta_LTotal r500 Delta_r500 M500 "
         "Delta_M500 M500_gas Delta_M500_gas M500_2D_star Delta_M500_2D_star M500_3D_star Delta_M500_3D_star"),
        "A0122 0.1134 3.65 0.15 0.84 0.03 2.57 0.16 0.89 0.03 2.26 0.19 1.98 0.21 0.68 0.04 0.55 0.03",
        "A1651 0.0845 6.10 0.25 0.87 0.09 3.11 0.22 1.18 0.03 5.15 0.42 6.70 0.32 0.82 0.06 0.65 0.05",
        "A2401 0.0571 2.06 0.07 0.33 0.01 1.30 0.09 0.68 0.02 0.95 0.10 0.85 0.09 0.35 0.03 0.27 0.02",
        "A2721 0.1144 4.78 0.23 0.57 0.01 2.81 0.21 1.03 0.03 3.46 0.32 4.36 0.57 0.74 0.06 0.57 0.04",
        "A2811 0.1079 4.89 0.20 0.85 0.14 2.13 0.18 1.04 0.03 3.59 0.28 4.47 0.17 0.56 0.05 0.47 0.04",
        "A2955 0.0943 2.13 0.10 0.60 0.03 1.33 0.08 0.68 0.04 0.99 0.11 0.66 0.05 0.35 0.02 0.30 0.02",
        "A2984 0.1042 2.08 0.07 0.86 0.03 1.72 0.08 0.67 0.01 0.95 0.10 1.05 0.08 0.46 0.02 0.39 0.02",
        "A3112 0.0750 4.54 0.11 0.93 0.05 3.33 0.23 1.02 0.02 3.23 0.19 4.29 0.16 0.88 0.06 0.70 0.04",
        "A3693 0.1237 3.63 0.20 0.71 0.07 2.42 0.18 0.90 0.03 2.26 0.23 2.49 0.15 0.64 0.05 0.51 0.04",
        "A4010 0.0963 3.78 0.13 0.81 0.12 2.65 0.21 0.92 0.02 2.41 0.18 2.87 0.11 0.70 0.06 0.56 0.05",
        "AS0084 0.1100 3.75 0.20 0.70 0.02 2.46 0.16 0.91 0.03 2.37 0.24 2.09 0.16 0.65 0.04 0.52 0.03",
        "AS0296 0.0696 2.70 0.21 0.57 0.01 1.30 0.07 0.78 0.04 1.45 0.21 1.09 0.10 0.35 0.02 0.29 0.01",
        "A0478 0.0881 7.09 0.12 none none none none 1.28 0.03 6.58 0.38 11.5 0.8 none none none none",
        "A2029 0.0773 8.41 0.12 none none none none 1.42 0.03 8.71 0.55 12.0 0.4 none none none none",
        "A2390 0.2329 10.6 0.8 none none none none 1.50 0.07 11.8 1.8 17.2 1.0 none none none none",
    ]

    table_fractions = [
        "ClusterID fgas Delta_fgas fstar Delta_fstar fbaryons Delta_fbaryons",
        "A0122 0.088 0.012 0.024 0.002 0.112 0.012",
        "A1651 0.130 0.012 0.013 0.001 0.143 0.012",
        "A2401 0.089 0.013 0.028 0.003 0.118 0.014",
        "A2721 0.126 0.020 0.017 0.002 0.143 0.020",
        "A2811 0.125 0.011 0.013 0.002 0.138 0.011",
        "A2955 0.067 0.010 0.030 0.004 0.097 0.011",
        "A2984 0.111 0.014 0.041 0.005 0.152 0.015",
        "A3112 0.133 0.009 0.022 0.002 0.155 0.009",
        "A3693 0.110 0.013 0.023 0.003 0.133 0.013",
        "A4010 0.119 0.010 0.023 0.003 0.143 0.010",
        "AS0084 0.088 0.011 0.022 0.003 0.110 0.011",
        "AS0296 0.075 0.013 0.020 0.003 0.095 0.013",
        "A0478 0.173 0.016 none none none none",
        "A2029 0.130 0.009 none none none none",
        "A2390 0.144 0.023 none none none none",
    ]

    def __init__(self, *args, **kwargs):
        super(Gonzalez13, self).__init__(*args, **kwargs)
        self.process_data()

    def process_data(self):
        # Construct Pandas dataset with cluster names as indices
        table_series = pd.Series(self.table_masses).str.split(' ')
        table_series = np.array(table_series.values.tolist())
        df_masses = pd.DataFrame(table_series[1:, 1:], columns=table_series[0, 1:], index=table_series[1:, 0])
        df_masses = df_masses.apply(pd.to_numeric, errors='ignore', downcast='float')

        table_series = pd.Series(self.table_fractions).str.split(' ')
        table_series = np.array(table_series.values.tolist())
        df_fractions = pd.DataFrame(table_series[1:, 1:], columns=table_series[0, 1:], index=table_series[1:, 0])
        df_fractions = df_fractions.apply(pd.to_numeric, errors='ignore', downcast='float')

        h_conv_Gonz = cosmology.WMAP7.h / self.cosmo_model.h

        print(df_masses, df_fractions)


class Barnes17(Observations):
    paper_name = "Barnes et al. (2017)"
    notes = "Assumes Planck13. Values for snapshots at z=0.1."

    field_names = [
        "Cluster name",
        "log10(M_500true)",
        "log10(M_500hse)",
        "log10(M_500spec)",
        "r_500true (Mpc)",
        "r_500spec (Mpc)",
        "IC location x (Mpc/h)",
        "IC location y (Mpc/h)",
        "IC location z (Mpc/h)",
        "IC extent (Mpc/h)",
        "k_B T_X (keV)",
        "L_X^{0.5−2.0 keV} (log10(L/erg s−1))",
        "M_gas <r_500spec (log10(M/M))",
        "M_star <r_500spec (log10(M/M))",
        "Y_X <r_500spec (log10(Y/MSun keV))",
        "Y_SZ <5*r_500spec (log10(Y/Mpc^2))",
        "Z_Fe (Z_FeSun)",
        "E_kin/E_thrm"
    ]

    table_Barn = [
        "CE-00 13.905 13.756 13.751 0.65 0.58 207.81 1498.48 1793.66 22.45 1.81 43.234 12.881 12.230 13.125 −5.445 0.26 0.11"
        "CE-01 13.982 13.927 13.931 0.69 0.66 1765.99 1721.52 1541.50 21.03 2.18 43.459 13.026 12.290 13.302 −5.521 0.21 0.24"
        "CE-02 13.920 13.870 13.871 0.66 0.63 1962.55 1953.81 234.28 21.03 1.90 43.227 12.954 12.164 13.217 −5.425 0.27 0.05"
        "CE-03 13.926 13.877 13.861 0.66 0.63 1772.74 1915.43 616.81 27.33 2.01 43.114 12.924 12.140 13.221 −5.298 0.21 0.06"
        "CE-04 13.853 13.723 13.697 0.62 0.55 1170.49 1524.83 1807.16 23.97 1.79 42.888 12.789 11.994 13.027 −5.449 0.22 0.11"
        "CE-05 13.892 13.902 13.880 0.64 0.64 395.90 613.11 1126.81 23.97 1.75 43.240 12.970 12.154 13.213 −5.340 0.29 0.09"
        "CE-06 14.128 14.102 14.083 0.77 0.74 1774.60 1519.13 206.88 35.53 2.28 43.518 13.204 12.306 13.551 −5.046 0.22 0.09"
        "CE-07 14.176 14.127 14.121 0.80 0.77 856.66 1662.35 869.10 31.16 2.48 43.657 13.247 12.354 13.631 −4.984 0.21 0.09"
        "CE-08 14.070 13.980 13.941 0.74 0.67 329.85 497.42 247.39 24.63 2.03 43.518 13.120 12.247 13.412 −5.126 0.25 0.09"
        "CE-09 14.244 14.118 14.084 0.84 0.74 922.84 982.93 1499.01 24.77 2.60 43.620 13.237 12.434 13.641 −4.850 0.20 0.16"
        "CE-10 14.301 14.250 14.225 0.88 0.83 1771.02 1093.98 1271.48 25.60 3.00 43.890 13.414 12.479 13.886 −4.758 0.26 0.05"
        "CE-11 14.290 14.182 14.182 0.87 0.80 1740.71 459.04 920.32 26.45 2.66 44.080 13.389 12.482 13.815 −4.845 0.29 0.21"
        "CE-12 14.422 14.349 14.326 0.96 0.90 793.25 945.74 682.99 36.72 3.28 43.942 13.485 12.644 13.990 −4.636 0.26 0.08"
        "CE-13 14.341 14.249 14.266 0.91 0.86 682.60 1023.43 1327.07 32.20 3.20 44.117 13.443 12.460 13.917 −4.721 0.29 0.06"
        "CE-14 14.563 14.571 14.568 1.08 1.08 184.91 991.14 1381.74 49.32 3.99 44.321 13.704 12.709 14.302 −4.508 0.23 0.31"
        "CE-15 14.407 14.404 14.418 0.95 0.96 1364.14 501.66 1179.09 41.86 3.56 44.037 13.511 12.592 13.993 −4.608 0.16 0.22"
        "CE-16 14.311 14.170 14.143 0.89 0.78 487.63 1526.81 406.49 41.86 3.06 43.877 13.351 12.505 13.814 −4.522 0.26 0.12"
        "CE-17 14.537 14.461 14.433 1.05 0.97 143.61 1251.36 1953.15 31.16 3.91 44.284 13.623 12.684 14.208 −4.525 0.21 0.25"
        "CE-18 14.639 14.587 14.555 1.14 1.07 542.69 580.55 1091.34 41.86 4.22 44.402 13.744 12.787 14.359 −4.262 0.19 0.09"
        "CE-19 14.586 14.483 14.445 1.09 0.98 547.19 219.86 760.69 37.94 3.23 44.102 13.660 12.723 14.171 −4.354 0.25 0.30"
        "CE-20 14.482 14.361 14.342 1.01 0.91 1827.41 1204.77 2009.96 30.16 3.60 44.127 13.551 12.660 14.077 −4.457 0.24 0.12"
        "CE-21 14.800 15.012 14.934 1.29 1.43 767.31 607.95 650.43 40.51 5.09 44.540 13.990 13.024 14.694 −4.009 0.51 0.29"
        "CE-22 14.837 14.721 14.701 1.33 1.20 1407.02 1572.74 567.84 62.05 6.04 44.618 13.937 12.967 14.668 −3.854 0.16 0.12"
        "CE-23 14.426 14.291 14.256 0.97 0.85 1378.96 2033.10 1838.00 37.94 3.16 43.904 13.452 12.642 13.934 −4.325 0.19 0.26"
        "CE-24 14.821 14.728 14.666 1.31 1.16 209.13 668.70 1948.65 52.66 5.31 44.412 13.876 12.937 14.598 −4.005 0.17 0.13"
        "CE-25 15.045 15.070 15.068 1.56 1.58 697.69 861.16 860.76 49.32 7.95 45.021 14.214 13.177 15.102 −3.650 0.23 0.31"
        "CE-26 14.899 14.838 14.780 1.39 1.27 1907.88 852.55 1346.40 43.26 6.43 44.589 14.010 13.037 14.809 −3.765 0.19 0.08"
        "CE-27 14.689 14.496 14.389 1.18 0.94 1790.48 618.80 1783.99 41.86 5.28 42.987 13.094 12.323 13.811 −4.259 0.19 0.24"
        "CE-28 14.902 14.688 14.671 1.39 1.17 944.94 707.08 1388.09 62.05 6.29 44.407 13.906 13.051 14.697 −3.735 0.18 0.13"
        "CE-29 15.077 15.089 14.912 1.60 1.40 719.79 1449.38 1015.75 60.04 7.66 44.942 14.188 13.185 15.067 −3.510 0.52 0.30"
    ]

    def __init__(self, *args, **kwargs):
        super(Barnes17, self).__init__(*args, **kwargs)
        self.process_data()

    def process_data(self):
        # Some `-` chars may not actually be minuses, replace those
        self.table_Barn = [re.sub(r'[^\x00-\x7F]+', '-', s) for s in self.table_Barn]

        # Construct Pandas dataset with cluster names as indices
        table_series = pd.Series(self.table_Barn).str.split(' ')
        table_series = np.array(table_series.values.tolist())
        df = pd.DataFrame(table_series[:, 1:], columns=self.field_names[1:], index=table_series[:, 0])

        # Convert all columns of DataFrame
        df = df.apply(pd.to_numeric, errors='ignore', downcast='float')

        h_conv_Barn = cosmology.Planck13.h / self.cosmo_model.h

        self.M_500true = np.power(10, df["log10(M_500true)"].to_numpy(dtype=np.float64)) * Solar_Mass
        self.M_500hse = np.power(10, df["log10(M_500hse)"].to_numpy(dtype=np.float64)) * Solar_Mass
        self.M_500spec = np.power(10, df["log10(M_500spec)"].to_numpy(dtype=np.float64)) * Solar_Mass
        self.r_500true = df["r_500true (Mpc)"].to_numpy(dtype=np.float64) * Mpc
        self.r_500spec = df["r_500spec (Mpc)"].to_numpy(dtype=np.float64) * Mpc
        self.x_ICs = df["IC location x (Mpc/h)"].to_numpy(dtype=np.float64) * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.y_ICs = df["IC location y (Mpc/h)"].to_numpy(dtype=np.float64) * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.z_ICs = df["IC location z (Mpc/h)"].to_numpy(dtype=np.float64) * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.extent_ICs = df["IC extent (Mpc/h)"].to_numpy(dtype=np.float64) * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.kB_TX = df["k_B T_X (keV)"].to_numpy(dtype=np.float64) * keV
        self.LX = np.power(10, df["L_X^{0.5−2.0 keV} (log10(L/erg s−1))"].to_numpy(dtype=np.float64)) * erg / second
        self.M_gas = np.power(10, df["M_gas <r_500spec (log10(M/M))"].to_numpy(dtype=np.float64)) * Solar_Mass
        self.M_star = np.power(10, df["M_star <r_500spec (log10(M/M))"].to_numpy(dtype=np.float64)) * Solar_Mass
        self.Y_X = np.power(10, df["Y_X <r_500spec (log10(Y/MSun keV))"].to_numpy(dtype=np.float64)) * Solar_Mass * keV
        self.Y_SZ = np.power(10, df["Y_SZ <5*r_500spec (log10(Y/Mpc^2))"].to_numpy(dtype=np.float64)) * Mpc ** 2
        self.Z_Fe = df["Z_Fe (Z_FeSun)"].to_numpy(dtype=np.float64) * Solar_Metallicity
        self.Ekin_Ethrm = df["E_kin/E_thrm"].to_numpy(dtype=np.float64) * Dimensionless

        # Convert self.LX to Solar luminosities
        self.LX = self.LX.to(Solar_Luminosity)

        # Return X-ray temperatures in Kelvin
        self.TX = (self.kB_TX / unyt.boltzmann_constant).to(K)

        # TODO: Review how h_conv_Barn is applied to each individual dataset


class Voit05(Observations):

    paper_name = "Voit et al. (2005)"
    notes = (
        "Dimensionless entropy profiles. "
        "The first set of simulated non-radiative clusters we will consider"
        "was produced by the entropy-conserving version of the SPH code"
        "GADGET (Springel, Yoshida & White 2001; Springel & Hernquist"
        "2002) with Omega_M = 0.3, Omega_ = 0.7, Omega_ = 0.045, h = 0.7, and"
        "σ 8 = 0.9. Most of this set comes from the non-radiative simulation"
        "described in Kay (2004), from which we take the 30 most massive clusters, "
        "ranging from 2.1 × 1013 to 7.5 × 1014 h−1 M_Sun."
    )

    radial_range_r200c = np.array([0.2, 1])

    a = 1.51
    b = 1.24

    def __init__(self, *args, **kwargs):
        super(Voit05, self).__init__(*args, **kwargs)

        self.k_k200c = 10 ** (np.log10(self.a) + self.b * np.log10(self.radial_range_r200c))
        r500c_r200c = 0.1 ** (1 / 3)
        self.radial_range_r500c = self.radial_range_r200c * r500c_r200c
        self.k_k500c = 10 ** (np.log10(self.a) + self.b * np.log10(self.radial_range_r500c))


