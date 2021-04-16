import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import itertools

from .cosmology import Article, repository_dir

comment = (
    "Planck 2015 results. XXVII. The second Planck catalogue of Sunyaev-Zeldovich sources"
)


class PlanckSZ2015(Article):

    field_names = ('source_number name y5r500 y5r500_error validation_status redshift redshift_source_name '
                   'mass_sz mass_sz_pos_err mass_sz_neg_err').split()

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Planck Collaboration (2015)",
            comment=comment,
            bibcode="2016A&A...594A..27P",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..27P/abstract",
            **cosmo_kwargs
        )
        self.hconv = 0.70 / self.h
        self.process_data()
        self.bin_data()

    def process_data(self):

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
            ez ** (-2 / 3) * (luminosity_distance.value * self.hconv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            ez ** (-2 / 3) * (luminosity_distance.value * self.hconv * (np.pi / 10800.0) * unyt.arcmin) ** 2.0,
            1,
            1,
            None,
            1.e14 * self.hconv * unyt.Solar_Mass,
            1.e14 * self.hconv * unyt.Solar_Mass,

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

        setattr(self, 'binned_mass_sz', np.asarray(bin_centres) * unyt.Solar_Mass)
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