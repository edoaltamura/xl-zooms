import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Halo mass - Gas fraction relation from 43 low-redshift Chandra-observed groups."
    "Gas and total mass profiles for 23 low-redshift, relaxed groups spanning "
    "a temperature range 0.7-2.7 keV, derived from Chandra. "
    "Data was corrected for the simulation's cosmology. Gas fraction (<R_500)."
)


class Sun2009(Article):
    data_files: List[str] = [
        os.path.join(repository_dir, 'Sun2009.dat'),
        os.path.join(repository_dir, 'Sun2009_table1.dat'),
        os.path.join(repository_dir, 'Sun2009_table3.dat'),
        os.path.join(repository_dir, 'Sun2009_table4.dat'),
        os.path.join(repository_dir, 'Sun09_entropy_profiles.txt'),
    ]

    def __init__(self, reduced_table: bool = False, disable_cosmo_conversion: bool = False, **cosmo_kwargs):
        super().__init__(
            citation="Sun et al. (2009)",
            comment=comment,
            bibcode="2009ApJ...693.1142S",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2009ApJ...693.1142S/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.73 / self.h
        if disable_cosmo_conversion:
            self.hconv = 1.

        if reduced_table:
            self.get_reduced_table()
        else:
            self.get_tab1()
            self.get_tab3()
            self.get_tab4()

    def get_reduced_table(self):
        raw = np.loadtxt(self.data_files[0])
        z = raw[:, 0]
        M_500 = unyt.unyt_array((10 ** 13) * self.hconv * raw[:, 1], units="Msun")
        error_M_500_p = unyt.unyt_array((10 ** 13) * self.hconv * raw[:, 2], units="Msun")
        error_M_500_m = unyt.unyt_array((10 ** 13) * self.hconv * raw[:, 3], units="Msun")

        fb_500 = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 4], units="dimensionless")
        error_fb_500_p = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 5], units="dimensionless")
        error_fb_500_m = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 6], units="dimensionless")

        fb_2500 = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 7], units="dimensionless")
        error_fb_2500_p = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 8], units="dimensionless")
        error_fb_2500_m = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 9], units="dimensionless")

        K_500 = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 10], units="keV*cm**2")
        error_K_500_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 11], units="keV*cm**2")
        error_K_500_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 12], units="keV*cm**2")

        K_1000 = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 13], units="keV*cm**2")
        error_K_1000_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 14], units="keV*cm**2")
        error_K_1000_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 15], units="keV*cm**2")

        K_1500 = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 16], units="keV*cm**2")
        error_K_1500_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 17], units="keV*cm**2")
        error_K_1500_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 18], units="keV*cm**2")

        K_2500 = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 19], units="keV*cm**2")
        error_K_2500_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 20], units="keV*cm**2")
        error_K_2500_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 21], units="keV*cm**2")

        K_0p15r500 = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 22], units="keV*cm**2")
        error_K_0p15r500_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 23], units="keV*cm**2")
        error_K_0p15r500_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 24], units="keV*cm**2")

        K_30kpc = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 25], units="keV*cm**2")
        error_K_30kpc_p = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 26], units="keV*cm**2")
        error_K_30kpc_m = unyt.unyt_array(self.hconv ** (1 / 3) * raw[:, 27], units="keV*cm**2")

        R_500 = unyt.unyt_array(self.hconv * raw[:, 28], units="kpc")
        error_R_500_p = unyt.unyt_array(self.hconv * raw[:, 29], units="kpc")
        error_R_500_m = unyt.unyt_array(self.hconv * raw[:, 30], units="kpc")

        R_2500 = unyt.unyt_array(self.hconv * raw[:, 31], units="kpc")
        error_R_2500_p = unyt.unyt_array(self.hconv * raw[:, 32], units="kpc")
        error_R_2500_m = unyt.unyt_array(self.hconv * raw[:, 33], units="kpc")

        # Class parser
        # Define the scatter as offset from the mean value
        self.M_500 = M_500
        self.M_500_error = unyt.unyt_array((error_M_500_m, error_M_500_p))

        self.fb_500 = fb_500
        self.fb_500_error = unyt.unyt_array((error_fb_500_m, error_fb_500_p))

        self.fb_2500 = fb_2500
        self.fb_2500_error = unyt.unyt_array((error_fb_2500_m, error_fb_2500_p))

        self.K_500 = K_500
        self.K_500_error = unyt.unyt_array((error_K_500_m, error_K_500_p))

        self.K_1000 = K_1000
        self.K_1000_error = unyt.unyt_array((error_K_1000_m, error_K_1000_p))

        self.K_1500 = K_1500
        self.K_1500_error = unyt.unyt_array((error_K_1500_m, error_K_1500_p))

        self.K_2500 = K_2500
        self.K_2500_error = unyt.unyt_array((error_K_2500_m, error_K_2500_p))

        self.K_0p15r500 = K_0p15r500
        self.K_0p15r500_error = unyt.unyt_array((error_K_0p15r500_m, error_K_0p15r500_p))

        self.K_30kpc = K_30kpc
        self.K_30kpc_error = unyt.unyt_array((error_K_30kpc_m, error_K_30kpc_p))

        self.R_500 = R_500
        self.R_500_error = unyt.unyt_array((error_R_500_m, error_R_500_p))

        self.R_2500 = R_2500
        self.R_2500_error = unyt.unyt_array((error_R_2500_m, error_R_2500_p))

        self.M_500gas = M_500 * fb_500
        self.M_500gas_error = unyt.unyt_array((
            self.M_500gas * (error_M_500_m / M_500 + error_fb_500_m / fb_500),
            self.M_500gas * (error_M_500_p / M_500 + error_fb_500_p / fb_500)
        ))

        self.K_500_adi = unyt.unyt_quantity(342, "keV*cm**2") * \
                         (M_500 / unyt.unyt_quantity(1e14, "Msun")) ** (2 / 3) * \
                         self.hconv ** (-4 / 3) * self.ez_function(z) ** (-2 / 3) * \
                         (0.165 / self.fb0) ** (-2 / 3)

        self.R_1000 = R_500 * 0.741
        self.R_1000_error = unyt.unyt_array((
            self.R_1000 * (error_R_500_m / R_500 + 0.013 / 0.741),
            self.R_1000 * (error_R_500_p / R_500 + 0.013 / 0.741)
        ))

        self.R_1500 = R_500 * 0.617
        self.R_1500_error = unyt.unyt_array((
            self.R_1500 * (error_R_500_m / R_500 + 0.011 / 0.617),
            self.R_1500 * (error_R_500_p / R_500 + 0.011 / 0.617)
        ))

    @staticmethod
    def load_table(file: str):

        data = []
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            if not (
                    line.startswith('#') or
                    line.startswith('\t') or
                    line.isspace()
            ):
                line_data = line.strip().split('\t')
                data.append(line_data)

        return data

    def get_tab1(self):
        data = self.load_table(self.data_files[1])

        self.group_name: List[str] = []
        self.redshift: List[float] = []

        for line in data:
            self.group_name.append(line[0])
            self.redshift.append(float(line[1]))

        self.group_name = np.array(self.group_name, dtype=str)
        self.redshift = unyt.unyt_array(np.array(self.redshift), 'dimensionless')

    def get_tab3(self):
        data = self.load_table(self.data_files[2])

        fields = ['T_500', 'T_2500', 'r_500', 'r_2500', 'M_500', 'f_gas500']
        units = ['keV', 'keV', 'kpc', 'kpc', '1e13*Msun', 'dimensionless']
        col_index = [1, 2, 3, 4, 5, 6]
        cosmo_conv = [1 / self.hconv, 1 / self.hconv, self.hconv, self.hconv, self.hconv ** 1.5]

        for field, unit, col, conv in zip(fields, units, col_index, cosmo_conv):

            collector = []
            for line in data:
                collector.append(line[col])

            for i, t in enumerate(collector):
                t = re.sub('[()*]', '', t)
                if '+or-' in t:
                    t = t.split('+or-')[0]
                elif '^' in t and '_' in t:
                    t = t.split('^')[0]
                t = t.strip()
                if t == '':
                    collector[i] = np.nan
                else:
                    collector[i] = float(t)

            collector = unyt.unyt_array(np.array(collector) * conv, unit)
            setattr(self, field, collector)

        # Reconstruct r_1000 and r_1500 from the scaling-radii relations
        # For tier 1 & 2 (those with M_500 defined)
        # r1000 / r500 = 0.741 ± 0.013
        # r1500 / r500 = 0.617 ± 0.011
        not_tier_12 = np.isnan(self.M_500.value)
        self.r_1000 = self.r_500.copy() * 0.741
        self.r_1000[not_tier_12] = np.nan
        self.r_1500 = self.r_500.copy() * 0.617
        self.r_1500[not_tier_12] = np.nan

        self.M_500 = self.M_500.to('Msun')

    def get_tab4(self):
        data = self.load_table(self.data_files[3])

        fields = ['K_500', 'K_1000', 'K_1500', 'K_2500', 'K_0p15r500', 'K_30kpc']
        units = ['keV*cm**2'] * 6
        col_index = [1, 2, 3, 4, 5, 6]
        cosmo_conv = [self.hconv ** (1 / 3)] * 6

        for field, unit, col, conv, z in zip(fields, units, col_index, cosmo_conv, self.redshift):

            collector = []
            for line in data:
                collector.append(line[col])

            for i, t in enumerate(collector):
                t = re.sub('[()*]', '', t)
                if '+or-' in t:
                    t = t.split('+or-')[0]
                elif '^' in t and '_' in t:
                    t = t.split('^')[0]
                t = t.strip()
                if t == '':
                    collector[i] = np.nan
                else:
                    collector[i] = float(t) * self.ez_function(z) ** (4 / 3)

            collector = unyt.unyt_array(np.array(collector) * conv, unit)
            setattr(self, field, collector)

        # Reconstruct r_1000 from the scaling-radii relations
        # For tier 3 (those with K_1000 defined)
        # r1000 ∼ 0.73r500
        tier_3 = np.isnan(self.M_500) & ~np.isnan(self.K_1000)
        self.r_1000[tier_3] = self.r_500[tier_3] * 0.73

        # Reconstruct K_500, adiabatic (eq 7)
        self.K_500_adi = np.ones_like(self.K_500)
        for i, (m500, z) in enumerate(zip(self.M_500, self.redshift)):
            self.K_500_adi[i] = 342 * unyt.keV * unyt.cm ** 2 * (m500 / 1e14 / unyt.Solar_Mass) ** (2 / 3)
            self.K_500_adi[i] *= (0.165 / self.fb0) ** (-2 / 3)
            self.K_500_adi[i] *= self.ez_function(z) ** (-2 / 3)
            self.K_500_adi[i] *= self.hconv ** (-4 / 3)

    def filter_by(self, selection_field: str, low: float, high: float):
        selection_data = getattr(self, selection_field).value
        fields = ['T_500', 'T_2500', 'r_500', 'r_2500', 'M_500', 'K_500',
                  'K_1000', 'K_1500', 'K_2500', 'K_0p15r500', 'K_30kpc']
        print(f'[Literature] {self.citation} data filtered by {selection_field}:({low:.2E}->{high:.2E})')
        for i, selection_value in enumerate(selection_data):
            if np.isnan(selection_value):
                continue
            elif selection_value < low or selection_value > high:
                for field in fields:
                    dataset = getattr(self, field)
                    dataset[i] = np.nan
                    setattr(self, field, dataset)

    def get_shortcut(self):
        raw = np.loadtxt(self.data_files[4], delimiter=',')
        r_r500, S_S500_50, S_S500_10, S_S500_90 = raw.T
        return r_r500, S_S500_50, S_S500_10, S_S500_90

    def overlay_points(self, axes: plt.Axes, x: str, y: str, **kwargs) -> None:
        if axes is None:
            fig, axes = plt.subplots()
            axes.loglog()
            axes.set_xlabel(x)
            axes.set_ylabel(y)

        x = getattr(self, x)
        y = getattr(self, y)
        axes.scatter(x, y, **kwargs)
        plt.show()

    def overlay_entropy_profiles(
            self,
            axes: plt.Axes = None,
            r_units: str = 'r500',
            k_units: str = 'K500adi',
            vkb05_line: bool = True,
            color: str = 'k',
            alpha: float = 1.,
            markersize: float = 1,
            linewidth: float = 0.5
    ) -> None:

        stand_alone = False
        if axes is None:
            stand_alone = True
            fig, axes = plt.subplots()
            axes.loglog()
            axes.set_xlabel(f'$r$ [{r_units}]')
            axes.set_ylabel(f'$K$ [${k_units}$]')
            axes.axvline(1, linestyle=':', color=color, alpha=alpha)

        # Set-up entropy data
        fields = ['K_500', 'K_1000', 'K_1500', 'K_2500', 'K_0p15r500', 'K_30kpc']
        K_stat = dict()
        if k_units == 'K500adi':
            K_conv = 1 / getattr(self, 'K_500_adi')
            axes.axhline(1, linestyle=':', color=color, alpha=alpha)
        elif k_units == 'keVcm^2':
            K_conv = np.ones_like(getattr(self, 'K_500_adi'))
            axes.fill_between(
                np.array(axes.get_xlim()),
                y1=np.nanmin(self.K_500_adi),
                y2=np.nanmax(self.K_500_adi),
                facecolor='k',
                alpha=0.3
            )
        else:
            raise ValueError("Conversion unit unknown.")
        for field in fields:
            data = np.multiply(getattr(self, field), K_conv)
            K_stat[field] = (
                np.nanpercentile(data, 16),
                np.nanpercentile(data, 50),
                np.nanpercentile(data, 84)
            )
            K_stat[field.replace('K', 'num')] = np.count_nonzero(~np.isnan(data))

        # Set-up radial distance data
        r_stat = dict()
        if r_units == 'r500':
            r_conv = 1 / getattr(self, 'r_500')
        elif r_units == 'r2500':
            r_conv = 1 / getattr(self, 'r_2500')
        elif r_units == 'kpc':
            r_conv = np.ones_like(getattr(self, 'r_2500'))
        else:
            raise ValueError("Conversion unit unknown.")
        for field in ['r_500', 'r_1000', 'r_1500', 'r_2500']:
            data = np.multiply(getattr(self, field), r_conv)
            if k_units == 'K500adi':
                data[np.isnan(self.K_500_adi)] = np.nan
            r_stat[field] = (
                np.nanpercentile(data, 16),
                np.nanpercentile(data, 50),
                np.nanpercentile(data, 84)
            )
            r_stat[field.replace('r', 'num')] = np.count_nonzero(~np.isnan(data))
        data = np.multiply(getattr(self, 'r_500') * 0.15, r_conv)
        if k_units == 'K500adi':
            data[np.isnan(self.K_500_adi)] = np.nan
        r_stat['r_0p15r500'] = (
            np.nanpercentile(data, 16),
            np.nanpercentile(data, 50),
            np.nanpercentile(data, 84)
        )
        r_stat['num_0p15r500'] = np.count_nonzero(~np.isnan(data))
        data = np.multiply(
            np.ones_like(getattr(self, 'r_2500')) * 30 * unyt.kpc,
            r_conv
        )
        if k_units == 'K500adi':
            data[np.isnan(self.K_500_adi)] = np.nan
        r_stat['r_30kpc'] = (
            np.nanpercentile(data, 16),
            np.nanpercentile(data, 50),
            np.nanpercentile(data, 84)
        )
        r_stat['num_30kpc'] = np.count_nonzero(~np.isnan(data))

        for suffix in ['_500', '_1000', '_1500', '_2500', '_0p15r500', '_30kpc']:
            x_low, x, x_hi = r_stat['r' + suffix]
            y_low, y, y_hi = K_stat['K' + suffix]
            num_objects = f"{r_stat['num' + suffix]}, {K_stat['num' + suffix]}"
            point_label = f"r{suffix:.<17s} Num(x,y) = {num_objects}"
            if stand_alone:
                axes.scatter(x, y, label=point_label, s=markersize)
                axes.errorbar(
                    x, y, yerr=[[y_hi - y], [y - y_low]], xerr=[[x_hi - x], [x - x_low]],
                    ls='none', ms=markersize, lw=linewidth
                )
            else:
                axes.scatter(x, y, color=color, alpha=alpha, s=markersize)
                axes.errorbar(
                    x, y, yerr=[[y_hi - y], [y - y_low]], xerr=[[x_hi - x], [x - x_low]],
                    ls='none', ecolor=color, alpha=alpha, ms=markersize, lw=linewidth
                )

        if vkb05_line:
            if r_units == 'r500' and k_units == 'K500adi':
                r = np.linspace(*axes.get_xlim(), 31)
                k = 1.40 * r ** 1.1 / self.hconv
                axes.plot(r, k, linestyle='--', color=color, alpha=alpha)
            else:
                print((
                    "The VKB05 adiabatic threshold should be plotted only when both "
                    "axes are in scaled units, since the line is calibrated on an NFW "
                    "profile with self-similar halos with an average concentration of "
                    "c_500 ~ 4.2 for the objects in the Sun et al. (2009) sample."
                ))

        if k_units == 'K500adi':
            r_r500, S_S500_50, S_S500_10, S_S500_90 = self.get_shortcut()

            plt.fill_between(
                r_r500,
                S_S500_10,
                S_S500_90,
                color='grey', alpha=0.5, linewidth=0
            )
            plt.plot(r_r500, S_S500_50, c='k')

        if stand_alone:
            plt.legend()
            plt.show()
