import re
import os
import unyt
import h5py
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "C-EAGLE"
    "Assumes Planck13. Values for snapshots at z=0.1."
)

class Barnes2017(Article):
    citation = "Barnes et al. (2017)"
    notes = ""

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Barnes et al. (2017)",
            comment=comment,
            bibcode="2017MNRAS.471.1088B",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.1088B/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.6777 / self.h

        self.process_data()
        self.get_from_hdf5()

    def process_data(self):

        data = np.loadtxt(f'{repository_dir}/barnes2017_ceagle_properties.dat').T

        self.m_500true = np.power(10, data[0]) * unyt.Solar_Mass
        self.m_500hse = np.power(10, data[1]) * unyt.Solar_Mass
        self.m_500spec = np.power(10, data[2]) * unyt.Solar_Mass
        self.r_500true = data[3] * unyt.Mpc
        self.r_500spec = data[4] * unyt.Mpc
        self.x_ics = data[5] * unyt.Mpc / self.h
        self.y_ics = data[6] * unyt.Mpc / self.h
        self.z_ics = data[7] * unyt.Mpc / self.h
        self.extent_ics = data[8] * unyt.Mpc / self.h
        self.kb_TX = data[9] * unyt.keV
        self.LX = np.power(10, data[10]) * unyt.erg / unyt.second * self.hconv ** 2
        self.m_gas = np.power(10, data[11]) * unyt.Solar_Mass
        self.m_star = np.power(10, data[12]) * unyt.Solar_Mass
        self.Y_X = np.power(10, data[13]) * unyt.Solar_Mass * unyt.keV
        self.Y_SZ = np.power(10, data[14]) * unyt.Mpc ** 2
        self.Z_Fe = data[15] * unyt.Solar_Metallicity
        self.ekin_ethrm = data[16] * unyt.Dimensionless

    def get_from_hdf5(self):
        data = self.load_dict_from_hdf5(f'{repository_dir}/barnes2017_ceagle.hdf5')
        data = self.dict2obj(data)
        self.hdf5 = data

    def load_dict_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')

    def recursively_load_dict_contents_from_group(self, h5file, path):
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file, path + key + '/')
        return ans

    def dict2obj(self, d):
        # Check if object d is an instance of class list
        if isinstance(d, list):
            d = [self.dict2obj(x) for x in d]
        if not isinstance(d, dict):
            return d

        class C:
            pass

        obj = C()
        for k in d:
            k_name = k.lower() if k == 'True' else k
            obj.__dict__[k_name] = self.dict2obj(d[k])

        return obj