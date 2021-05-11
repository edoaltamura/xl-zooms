import os
import numpy as np
from typing import Union, List
from astropy import cosmology
import unyt


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
repository_dir = os.path.join(os.path.dirname(__file__), 'repository')


class Cosmology(object):

    def __init__(self, cosmo_model: str = "Planck18", verbose: int = 1):

        self.verbose = verbose

        for model_name in dir(cosmology):
            if cosmo_model.lower() == model_name.lower():
                if self.verbose > 1:
                    print(f"Using the {model_name} cosmology")
                self.cosmo_model = getattr(cosmology, model_name)
                break

        self.h = self.cosmo_model.h
        self.Ob0 = self.cosmo_model.Ob0
        self.Om0 = self.cosmo_model.Om0
        self.fb = self.Ob0 / self.Om0


    def time_from_redshift(self, z: Union[float, List[float], np.ndarray]):
        """
        Assuming the current cosmology, takes a redshift value (or list or numpy array)
        and returns the corresponding age of the Universe at that redshift.
        """
        if isinstance(z, float):
            return self.cosmo_model.age(0) - self.cosmo_model.lookback_time(z)
        else:
            convert = [self.cosmo_model.age(0) - self.cosmo_model.lookback_time(redshift) for redshift in z]
            return np.asarray(convert)

    def redshift_from_time(self, t: Union[unyt.unyt_quantity, unyt.unyt_array]):
        """
        Assuming the current cosmology, takes a value (or list or numpy array)
        for the age of the Universe and returns the corresponding redshift.
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
        """
        if isinstance(z, unyt.unyt_quantity) or isinstance(z, float):
            return self.cosmo_model.efunc(z)
        else:
            convert = [self.cosmo_model.efunc(redshift) for redshift in z]
            return np.asarray(convert)

    def luminosity_distance(self, *args, **kwargs):

        return self.cosmo_model.luminosity_distance(*args, **kwargs)

    def age(self, *args, **kwargs) -> unyt.unyt_quantity:

        t = self.cosmo_model.age(*args, **kwargs)
        value = t.value
        units = str(t.unit)
        return unyt.unyt_quantity(value, units)



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
