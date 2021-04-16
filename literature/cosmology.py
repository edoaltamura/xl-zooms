import re
import os
import numpy as np
import h5py
import itertools
from typing import Union, List, Tuple
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


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def dict2obj(d):
    # Check if object d is an instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C:
        pass

    obj = C()
    for k in d:
        k_name = k.lower() if k == 'True' else k
        obj.__dict__[k_name] = dict2obj(d[k])

    return obj


class Cosmology(object):

    def __init__(self, cosmo_model: str = "Planck18", verbose: int = 1):

        self.verbose = verbose

        for model_name in dir(cosmology):
            if cosmo_model.lower() == model_name.lower():
                if self.verbose > 1:
                    print(f"Using the {model_name} cosmology")
                self.cosmo_model = getattr(cosmology, model_name)
                self.h = self.cosmo_model.h
                self.Ob0 = self.cosmo_model.Ob0
                self.Om0 = self.cosmo_model.Om0
                self.fb = self.Ob0 / self.Om0
                break

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


class Article(Cosmology):

    def __init__(self, citation: str, comment: str, bibcode: str, hyperlink: str,
                 **cosmo_kwargs):
        super().__init__(**cosmo_kwargs)

        self.citation = citation
        self.comment = comment
        self.bibcode = bibcode
        self.hyperlink = hyperlink

        if self.verbose > 0:
            print(f"[Literature data] {self.citation} --> {self.hyperlink}")


class Gonzalez13(Article):
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


class Barnes17(Article):
    citation = "Barnes et al. (2017) (C-EAGLE)"
    notes = "Assumes Planck13. Values for snapshots at z=0.1."

    def __init__(self, *args, **kwargs):
        super(Barnes17, self).__init__(*args, **kwargs)
        self.process_data()
        self.get_from_hdf5()

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
        self.LX = np.power(10, data[10]) * erg / second * h_conv_Barn ** 2
        self.m_gas = np.power(10, data[11]) * Solar_Mass
        self.m_star = np.power(10, data[12]) * Solar_Mass
        self.Y_X = np.power(10, data[13]) * Solar_Mass * keV
        self.Y_SZ = np.power(10, data[14]) * Mpc ** 2
        self.Z_Fe = data[15] * Solar_Metallicity
        self.ekin_ethrm = data[16] * Dimensionless

    def get_from_hdf5(self):
        data = load_dict_from_hdf5(f'{repository_dir}/barnes2017_ceagle.hdf5')
        data = dict2obj(data)
        self.hdf5 = data


class Voit05(Article):
    citation = "Voit et al. (2005)"
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



