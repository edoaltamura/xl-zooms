import re
import os
import numpy as np
import itertools
from typing import Union, List
from astropy import cosmology
from matplotlib import pyplot as plt
import unyt
from unyt import (
    Solar_Mass,
    Mpc,
    Dimensionless,
    Solar_Luminosity,
    Solar_Metallicity,
    keV, erg, second, K
)

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

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
        The only available cosmology types in this method are: FlatLambdaCDM,
        FlatwCDM, LambdaCDM and wCDM. See `astropy.cosmology`_ for more details on
        these types of cosmologies. To create a cosmology of a type that isn't
        listed above, it will have to be created directly using astropy.cosmology.

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

# === End of imports ===
unyt.define_unit("hubble_parameter", value=1. * Dimensionless, tex_repr="h")
repository_dir = os.path.join(os.path.dirname(__file__), 'repository')


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
                print(f"[Literature data] Applied: {self.citation}")
            except:
                pass

    def time_from_redshift(self, z: Union[float, List[float], np.ndarray]):
        """
        Assuming the current cosmology, takes a redshift value (or list or numpy array)
        and returns the corresponding age of the Universe at that redshift.
        In the case of lists, the calculation is mapped onto the input array in
        vectorized form.
        """
        if isinstance(z, float):
            return self.cosmo_model.age(0) - self.cosmo_model.lookback_time(z)
        else:
            convert = lambda redshift: (self.cosmo_model.age(0) - self.cosmo_model.lookback_time(redshift)).value
            return np.vectorize(convert)(np.asarray(z)) * self.cosmo_model.age(0).unit

    def redshift_from_time(self, t: Union[unyt.unyt_quantity, unyt.unyt_array]):
        """
        Assuming the current cosmology, takes a value (or list or numpy array)
        for the age of the Universe and returns the corresponding redshift.
        Note: numpy.vectorize is not a supported ufunc for unyt (astropy) arrays.
        """
        if isinstance(t, unyt.unyt_quantity):
            return self.cosmo_model.z_at_value(self.cosmo_model.age, t)
        else:
            convert = [self.cosmo_model.z_at_value(self.cosmo_model.age, time) for time in t]
            return np.asarray(convert)

    def ez_function(self, z: Union[unyt.unyt_quantity, unyt.unyt_array, float, np.ndarray]):
        """
        Assuming the current cosmology, takes a value (or list or numpy array)
        for the age of the Universe and returns the corresponding redshift.
        Note: numpy.vectorize is not a supported ufunc for unyt (astropy) arrays.
        """
        if isinstance(z, unyt.unyt_quantity) or isinstance(z, float):
            return self.cosmo_model.efunc(z)
        else:
            convert = [self.cosmo_model.efunc(redshift) for redshift in z]
            return np.asarray(convert)

    def luminosity_distance(self, *args, **kwargs):

        return self.cosmo_model.luminosity_distance(*args, **kwargs)


class Sun09(Observations):
    # Meta-data
    comment = (
        "Gas and total mass profiles for 23 low-redshift, relaxed groups spanning "
        "a temperature range 0.7-2.7 keV, derived from Chandra. "
        "Data was corrected for the simulation's cosmology. Gas fraction (<R_500)."
    )
    citation = "Sun et al. (2009)"
    bibcode = "2009ApJ...693.1142S"
    name = "Halo mass - Gas fraction relation from 43 low-redshift Chandra-observed groups."
    plot_as = "points"
    redshift = 0.1

    def __init__(self, *args, **kwargs):
        super(Sun09, self).__init__(*args, **kwargs)

        h_sim = self.cosmo_model.h
        Omega_b = self.cosmo_model.Ob0
        Omega_m = self.cosmo_model.Om0

        raw = np.loadtxt(f'{repository_dir}/Sun2009.dat')
        M_500 = unyt.unyt_array((10 ** 13) * (0.73 / h_sim) * raw[:, 1], units="Msun")
        error_M_500_p = unyt.unyt_array((10 ** 13) * (0.73 / h_sim) * raw[:, 2], units="Msun")
        error_M_500_m = unyt.unyt_array((10 ** 13) * (0.73 / h_sim) * raw[:, 3], units="Msun")
        fb_500 = unyt.unyt_array((0.73 / h_sim) ** 1.5 * raw[:, 4], units="dimensionless")
        error_fb_500_p = unyt.unyt_array((0.73 / h_sim) ** 1.5 * raw[:, 5], units="dimensionless")
        error_fb_500_m = unyt.unyt_array((0.73 / h_sim) ** 1.5 * raw[:, 6], units="dimensionless")

        # Class parser
        # Define the scatter as offset from the mean value
        self.M_500 = M_500
        self.M_500_error = unyt.unyt_array((error_M_500_m, error_M_500_p))
        self.fb_500 = fb_500
        self.fb_500_error = unyt.unyt_array((error_fb_500_m, error_fb_500_p))
        self.M_500gas = M_500 * fb_500
        self.M_500gas_error = unyt.unyt_array((
            self.M_500gas * (error_M_500_m / M_500 + error_fb_500_m / fb_500),
            self.M_500gas * (error_M_500_p / M_500 + error_fb_500_p / fb_500)
        ))


class Lovisari15(Observations):
    # Meta-data
    comment = (
        "23 Galaxy groups observed with XMM-Newton"
        "Corrected for the original cosmology which had h=0.7"
    )
    citation = "Lovisari et al. (2015)"
    bibcode = "2015A&A...573A.118L"
    name = "Scaling Properties of a Complete X-ray Selected Galaxy Group Sample"
    plot_as = "points"
    redshift = 0.1

    def __init__(self, *args, **kwargs):
        super(Lovisari15, self).__init__(*args, **kwargs)

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Lovisari2015.dat')
        self.M_500 = unyt.unyt_array((0.70 / self.cosmo_model.h) * 10 ** raw[:, 0], units="Msun")
        self.fb_500 = unyt.unyt_array(raw[:, 1] * (0.70 / self.cosmo_model.h) ** 2.5, units="dimensionless")
        self.M_gas500 = self.M_500 * self.fb_500


class Lin12(Observations):
    # Meta-data
    comment = (
        "Ionized gas out of 94 clusters combining Chandra, WISE and 2MASS. "
        "Data was corrected for the simulation's cosmology."
    )
    citation = "Lin et al. (2012; $z<0.25$ only)"
    bibcode = "2012ApJ...745L...3L"
    name = "Halo mass - Gas fraction relation from Chandra-observed clusters."
    plot_as = "points"
    redshift = 0.1

    def __init__(self, *args, **kwargs):
        super(Lin12, self).__init__(*args, **kwargs)

        h_sim = self.cosmo_model.h
        Omega_b = self.cosmo_model.Ob0
        Omega_m = self.cosmo_model.Om0

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Lin2012.dat')
        M_500 = unyt.unyt_array((0.71 / h_sim) * 10 ** raw[:, 0], units="Msun")
        M_500_error = unyt.unyt_array((0.71 / h_sim) * raw[:, 1], units="Msun")
        M_500_gas = unyt.unyt_array((0.71 / h_sim) * 10 ** raw[:, 2], units="Msun")
        M_500_gas_error = unyt.unyt_array((0.71 / h_sim) * raw[:, 3], units="Msun")
        z = raw[:, 6]

        # Compute the gas fractions
        fb_500 = (M_500_gas / M_500) * (0.71 / h_sim) ** 2.5
        fb_500_error = fb_500 * ((M_500_error / M_500) + (M_500_gas_error / M_500_gas))

        # Normalise by the cosmic mean
        # fb_500 = fb_500 / (Omega_b / Omega_m)
        # fb_500_error = fb_500_error / (Omega_b / Omega_m)

        # Select only the low-z data
        self.M_500 = M_500[z < 0.25]
        self.fb_500 = fb_500[z < 0.25]
        self.M_500gas = M_500_gas[z < 0.25]

        self.M_500_error = M_500_error[z < 0.25]
        self.fb_500_error = fb_500_error[z < 0.25]
        self.M_500gas_error = M_500_gas_error[z < 0.25]


class Eckert16(Observations):
    # Meta-data
    comment = (
        "Based on observations obtained with XMM-Newton. "
        "Corrected for the original cosmology which had h=0.7"
    )
    citation = "Eckert et al. (2016)"
    bibcode = "2016A&A...592A..12E"
    name = "The XXL Survey. XIII. Baryon content of the bright cluster sample"
    plot_as = "points"
    redshift = 0.1

    def __init__(self, *args, **kwargs):
        super(Eckert16, self).__init__(*args, **kwargs)

        h_sim = self.cosmo_model.h

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Eckert2016.dat')
        self.M_500 = unyt.unyt_array((0.70 / h_sim) * 10 ** raw[:, 0], units="Msun")
        self.fb_500 = unyt.unyt_array(raw[:, 1] * (0.70 / h_sim) ** 2.5, units="dimensionless")
        self.M_500gas = self.M_500 * self.fb_500


class Vikhlinin06(Observations):
    # Meta-data
    comment = (
        "Gas and total mass profiles for 13 low-redshift, relaxed clusters spanning "
        "a temperature range 0.7-9 keV, derived from all available Chandra data of "
        "sufficient quality. Data was corrected for the simulation's cosmology."
    )
    citation = "Vikhlinin et al. (2006)"
    bibcode = "2006ApJ...640..691V"
    name = "Halo mass - Gas fraction relation from 13 low-redshift Chandra-observed relaxed clusters."
    plot_as = "points"
    redshift = 0.1

    def __init__(self, *args, **kwargs):
        super(Vikhlinin06, self).__init__(*args, **kwargs)

        h_sim = self.cosmo_model.h

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Vikhlinin2006.dat')
        self.M_500 = unyt.unyt_array((10 ** 14) * (0.72 / h_sim) * raw[:, 1], units="Msun")
        self.error_M_500 = unyt.unyt_array((10 ** 14) * (0.72 / h_sim) * raw[:, 2], units="Msun")
        self.fb_500 = unyt.unyt_array((0.72 / h_sim) ** 1.5 * raw[:, 3], units="dimensionless")
        self.error_fb_500 = unyt.unyt_array((0.72 / h_sim) ** 1.5 * raw[:, 4], units="dimensionless")
        self.M_500gas = self.M_500 * self.fb_500
        self.error_M_500gas = self.M_500gas * (self.error_M_500 / self.M_500 + self.error_fb_500 / self.fb_500)


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
    hyperlink = 'https://ui.adsabs.harvard.edu/abs/2013ApJ...778...14G/abstract'
    notes = ("Data assumes WMAP7 as fiducial model. "
             "The quoted stellar baryon fractions include a deprojection correction. "
             "A0478, A2029, and A2390 are not part of the main sample. "
             "These clusters were included only in the X-ray analysis to extend the baseline to higher mass, but "
             "they have no photometry equivalent to the other systems with which to measure the stellar mass. "
             "At the high mass end, however, the stellar component contributes "
             "a relatively small fraction of the total baryons. "
             "The luminosities include appropriate e + k corrections for each galaxy from GZZ07. "
             "The stellar masses are quoted as observed, with no deprojection correction applied")

    data_fields = ('ClusterID redshift TX2 Delta_TX2 L_BCG_ICL Delta_L_BCG_ICL LTotal Delta_LTotal r500 Delta_r500 M500'
                   'Delta_M500 M500_gas Delta_M500_gas M500_2D_star Delta_M500_2D_star M500_3D_star Delta_M500_3D_star'
                   'fgas Delta_fgas fstar Delta_fstar fbaryons Delta_fbaryons').split()

    def __init__(self, *args, **kwargs):
        super(Gonzalez13, self).__init__(*args, **kwargs)
        self.process_data()

    def process_data(self):
        h_conv_Gonz = cosmology.WMAP7.h / self.cosmo_model.h

        conversion_factors = [
            0,
            1,
            keV,
            keV,
            1.e12 * h_conv_Gonz ** 2 * Solar_Luminosity,
            1.e12 * h_conv_Gonz ** 2 * Solar_Luminosity,
            1.e12 * h_conv_Gonz ** 2 * Solar_Luminosity,
            1.e12 * h_conv_Gonz ** 2 * Solar_Luminosity,
            h_conv_Gonz * Mpc,
            h_conv_Gonz * Mpc,
            1.e14 * h_conv_Gonz * Solar_Mass,
            1.e14 * h_conv_Gonz * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            1.e13 * h_conv_Gonz ** 2 * Solar_Mass,
            h_conv_Gonz * Dimensionless,
            h_conv_Gonz * Dimensionless,
            h_conv_Gonz * Dimensionless,
            h_conv_Gonz * Dimensionless,
            h_conv_Gonz * Dimensionless,
            h_conv_Gonz * Dimensionless,
        ]

        data = np.genfromtxt('repository/gonzalez2013.dat',
                             dtype=float,
                             invalid_raise=False,
                             missing_values='none',
                             usemask=False,
                             filling_values=np.nan).T

        for i, (field, conversion) in enumerate(zip(self.data_fields, conversion_factors)):
            setattr(self, field, data[i] * conversion)


class Barnes17(Observations):
    citation = "Barnes et al. (2017) (C-EAGLE)"
    notes = "Assumes Planck13. Values for snapshots at z=0.1."

    def __init__(self, *args, **kwargs):
        super(Barnes17, self).__init__(*args, **kwargs)
        self.process_data()

    def process_data(self):
        h_conv_Barn = cosmology.Planck13.h / self.cosmo_model.h
        data = np.loadtxt(f'{repository_dir}/barnes2017_ceagle_properties.dat').T

        self.m_500true = np.power(10, data[0]) * Solar_Mass
        self.m_500hse = np.power(10, data[1]) * Solar_Mass
        self.m_500spec = np.power(10, data[2]) * Solar_Mass
        self.r_500true = data[3] * Mpc
        self.r_500spec = data[4] * Mpc
        self.x_ics = data[5] * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.y_ics = data[6] * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.z_ics = data[7] * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.extent_ics = data[8] * h_conv_Barn * Mpc / unyt.hubble_parameter
        self.kb_TX = data[9] * keV
        self.LX = np.power(10, data[10]) * erg / second
        self.m_gas = np.power(10, data[11]) * Solar_Mass
        self.m_star = np.power(10, data[12]) * Solar_Mass
        self.Y_X = np.power(10, data[13]) * Solar_Mass * keV
        self.Y_SZ = np.power(10, data[14]) * Mpc ** 2
        self.Z_Fe = data[15] * Solar_Metallicity
        self.ekin_ethrm = data[16] * Dimensionless

        # Convert self.LX to Solar luminosities
        self.LX = self.LX.to(Solar_Luminosity)


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
        self.radial_range_r500c = self.radial_range_r200c / r500c_r200c
        self.k_k500c = 10 ** (np.log10(self.a) + self.b * np.log10(self.radial_range_r500c))

    def plot_on_axes(self, ax, **kwargs):
        r500c_r200c = 0.1 ** (1 / 3)
        radial_range_r500c = np.array(ax.get_xlim()) - np.log10(r500c_r200c)
        k_k500c = 10 ** (np.log10(self.a) + self.b * 10 ** radial_range_r500c)
        ax.plot(ax.get_xlim(), k_k500c, label=self.paper_name, **kwargs)


class Pratt10_(Observations):
    paper_name = "Pratt et al. (2010)"
    notes = (
        "Dimensionless entropy profiles. "
    )

    radial_range_r500c = np.array([0.5, 2])

    a = 1.42
    b = 1.1

    def __init__(self, *args, **kwargs):
        super(Pratt10_, self).__init__(*args, **kwargs)

        self.k_k500c = 10 ** (np.log10(self.a) + self.b * np.log10(self.radial_range_r500c))

    def plot_on_axes(self, ax, **kwargs):
        k_k500c = 10 ** (np.log10(self.a) + self.b * np.log10(np.array(ax.get_xlim())))
        ax.plot(ax.get_xlim(), k_k500c, label=self.paper_name, **kwargs)


class Pratt10(Observations):
    paper_name = "Pratt et al. (2010)"
    hyperlink = 'https://ui.adsabs.harvard.edu/abs/2010A%26A...511A..85P/abstract'
    notes = (
        "REXCESS sample. Entropy properties."
    )

    def __init__(self, *args, **kwargs):
        super(Pratt10, self).__init__(*args, **kwargs)

        self.process_properties()

    def process_properties(self):
        h_conv = 0.7 / self.cosmo_model.h
        field_names = (
            'Cluster_name z kT M500 Delta_hi_M500 Delta_lo_M500 K0p1R200 Delta_hi_K0p1R200 Delta_lo_K0p1R200 '
            'KR2500 Delta_KR2500 KR1000 Delta_KR1000 KR500 CC Disturbed').split()
        data = []

        with open('repository/pratt2018_properties.dat') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#') and not line.isspace():
                    line_data = line.split()
                    for i, element_data in enumerate(line_data):
                        if element_data.strip() == 'none':
                            # If no data, replace with Nan
                            line_data[i] = np.nan
                        elif re.search('[a-df-zA-Z]', element_data):
                            # If contains letters, remove white spaces
                            line_data[i] = element_data.strip()
                        else:
                            line_data[i] = float(element_data.strip())
                    data.append(line_data)

        data = list(map(list, itertools.zip_longest(*data, fillvalue=None)))
        for i, field in enumerate(data):
            data[i] = np.array(field)

        # Redshift columns: data[5]
        ez = self.ez_function(data[1])
        luminosity_distance = self.luminosity_distance(data[1]) / ((data[1] + 1) ** 2)

        conversion_factors = [
            None,
            1,
            ez ** (-2 / 3) * (luminosity_distance.value * h_conv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            ez ** (-2 / 3) * (luminosity_distance.value * h_conv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            1,
            1,
            None,
            1.e14 * h_conv * Solar_Mass,
            1.e14 * h_conv * Solar_Mass,

        ]

        for i, (field, conversion) in enumerate(zip(self.field_names, conversion_factors)):
            if isinstance(data[i][0], str):
                setattr(self, field, data[i])
            else:
                setattr(self, field, data[i] * conversion)

        print(data)


class PlanckSZ2015(Observations):
    paper_name = "Planck Collaboration (2015)"
    hyperlink = 'https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..27P/abstract'

    field_names = ('source_number name y5r500 y5r500_error validation_status redshift redshift_source_name '
                   'mass_sz mass_sz_pos_err mass_sz_neg_err').split()

    def __init__(self, *args, **kwargs):
        super(PlanckSZ2015, self).__init__(*args, **kwargs)
        self.process_data()
        self.bin_data()

    def process_data(self):
        h_conv = 0.7 / self.cosmo_model.h

        data = []

        with open(f'{repository_dir}/planck2015_sz2.dat') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#') and not line.isspace():
                    line_data = line.split('|')[1:-1]
                    for i, element_data in enumerate(line_data):
                        if element_data.isspace():
                            # If no data, replace with Nan
                            line_data[i] = np.nan
                        elif re.search('[a-df-zA-Z]', element_data):
                            # If contains letters, remove white spaces
                            line_data[i] = element_data.strip()
                        else:
                            line_data[i] = float(element_data.strip())
                    data.append(line_data)

        data = list(map(list, itertools.zip_longest(*data, fillvalue=None)))
        for i, field in enumerate(data):
            data[i] = np.array(field)

        # Redshift columns: data[5]
        ez = self.ez_function(data[5])
        luminosity_distance = self.luminosity_distance(data[5]) / ((data[5] + 1) ** 2)

        conversion_factors = [
            1,
            None,
            ez ** (-2 / 3) * (luminosity_distance.value * h_conv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            ez ** (-2 / 3) * (luminosity_distance.value * h_conv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            1,
            1,
            None,
            1.e14 * h_conv * Solar_Mass,
            1.e14 * h_conv * Solar_Mass,

        ]

        for i, (field, conversion) in enumerate(zip(self.field_names, conversion_factors)):
            if isinstance(data[i][0], str):
                setattr(self, field, data[i])
            else:
                setattr(self, field, data[i] * conversion)

    def bin_data(self, nbins: int = 10):
        bins = np.logspace(
            min(np.log10(self.mass_sz.value)),
            max(np.log10(self.mass_sz.value)),
            num=nbins
        )
        bin_centres = 10. ** (0.5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))
        digitized = np.digitize(self.mass_sz.value, bins)
        bin_median = [np.median(self.y5r500.value[digitized == i]) for i in range(1, len(bins))]
        bin_perc16 = [np.percentile(self.y5r500.value[digitized == i], 16) for i in range(1, len(bins))]
        bin_perc84 = [np.percentile(self.y5r500.value[digitized == i], 84) for i in range(1, len(bins))]

        setattr(self, 'binned_mass_sz', np.asarray(bin_centres) * Solar_Mass)
        setattr(self, 'binned_y5r500_median', np.asarray(bin_median) * unyt.arcmin ** 2)
        setattr(self, 'binned_y5r500_perc16', np.asarray(bin_perc16) * unyt.arcmin ** 2)
        setattr(self, 'binned_y5r500_perc84', np.asarray(bin_perc84) * unyt.arcmin ** 2)

        return bin_centres, bin_median, bin_perc16, bin_perc84

    def generate_kde(self):
        # Perform the kernel density estimate
        import scipy.stats as st
        x = np.log10(self.mass_sz.value[~np.isnan(self.mass_sz.value)])
        y = np.log10(self.y5r500.value[~np.isnan(self.y5r500.value)])
        xmin = min(np.log10(self.mass_sz.value))
        xmax = max(np.log10(self.mass_sz.value))
        ymin = min(np.log10(self.y5r500.value))
        ymax = max(np.log10(self.y5r500.value))
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        return 10 ** xx, 10 ** yy, f

    def quick_display(self):
        # Display the catalogue data
        plt.scatter(self.mass_sz.value, self.y5r500.value, c='orange', s=2)

        # kde = self.generate_kde()
        # plt.contour(kde[0], kde[1], kde[2], colors='k')

        # Overlay binned data
        plt.fill_between(
            self.binned_mass_sz,
            self.binned_y5r500_perc16,
            self.binned_y5r500_perc84,
            color='aqua', alpha=0.85, linewidth=0
        )
        plt.plot(self.binned_mass_sz, self.binned_y5r500_median, c='k')

        plt.ylabel(r'$Y_{SZ}\ (5 \times R_{500})$ [arcmin$^2$]')
        plt.xlabel(r'$M_{SZ}$ [M$_\odot$]')
        plt.title('Planck 2015 SZ2 catalogue')
        plt.xlim([5e13, 5e15])
        plt.ylim([1e-6, 2e-3])
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.close()


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

        hubble_parameter = Observations().cosmo_model.h

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


if __name__ == '__main__':
    obs = Pratt10()
