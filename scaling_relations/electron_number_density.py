from swiftsimio import SWIFTDataset
import numpy as np


# using https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4857P/abstract

def get_electron_number_density(sw_data: SWIFTDataset):
    ne_nH = np.zeros(len(sw_data.gas.element_mass_fractions.hydrogen.value)) + 1
    ni_nH = np.zeros(len(sw_data.gas.element_mass_fractions.hydrogen.value)) + 1
    mu = np.zeros(len(sw_data.gas.element_mass_fractions.hydrogen.value))

    # -- Sum element contributions
    # Hydrogen
    H = sw_data.gas.element_mass_fractions.hydrogen.value
    mu += 1.0 / (1.0 + 1.0)
    lN_H_AG = 12.00

    # Helium
    He_H = sw_data.gas.element_mass_fractions.helium.value / H
    ne_nH += (He_H) * (1.00794 / 4.002602) * (2.0 / 1.0)
    ni_nH += (He_H) * (1.00794 / 4.002602)
    mu += (He_H) / (1.0 + 2.0)
    AG_He = 10.99 - lN_H_AG
    He_H = 10.0 ** (np.log10(He_H * (1.00794 / 4.002602)) - AG_He)

    # Carbon
    C_H = sw_data.gas.element_mass_fractions.carbon.value / H
    ne_nH += (C_H) * (1.00794 / 12.0107) * (6.0 / 1.0)
    ni_nH += (C_H) * (1.00794 / 12.0107)
    mu += (C_H) / (1.0 + 6.0)
    AG_C = 8.56 - lN_H_AG
    C_H = 10.0 ** (np.log10(C_H * (1.00794 / 12.0107)) - AG_C)

    # Nitrogen
    N_H = sw_data.gas.element_mass_fractions.nitrogen.value / H
    ne_nH += (N_H) * (1.00794 / 14.0067) * (7.0 / 1.0)
    ni_nH += (N_H) * (1.00794 / 14.0067)
    mu += (N_H) / (1.0 + 7.0)
    AG_N = 8.05 - lN_H_AG
    N_H = 10.0 ** (np.log10(N_H * (1.00794 / 14.0067)) - AG_N)

    # Oxygen
    O_H = sw_data.gas.element_mass_fractions.oxygen.value / H
    ne_nH += (O_H) * (1.00794 / 15.9994) * (8.0 / 1.0)
    ni_nH += (O_H) * (1.00794 / 15.9994)
    mu += (O_H) / (1.0 + 8.0)
    AG_O = 8.93 - lN_H_AG
    O_H = 10.0 ** (np.log10(O_H * (1.00794 / 15.9994)) - AG_O)

    # Neon
    Ne_H = sw_data.gas.element_mass_fractions.neon.value / H
    ne_nH += (Ne_H) * (1.00794 / 20.1797) * (10.0 / 1.0)
    ni_nH += (Ne_H) * (1.00794 / 20.1797)
    mu += (Ne_H) / (1.0 + 10.0)
    AG_Ne = 8.09 - lN_H_AG
    Ne_H = 10.0 ** (np.log10(Ne_H * (1.00794 / 20.1797)) - AG_Ne)

    # Magnesium
    Mg_H = sw_data.gas.element_mass_fractions.magnesium.value / H
    ne_nH += (Mg_H) * (1.00794 / 24.3050) * (12.0 / 1.0)
    ni_nH += (Mg_H) * (1.00794 / 24.3050)
    mu += (Mg_H) / (1.0 + 12.0)
    AG_Mg = 7.58 - lN_H_AG
    Mg_H = 10.0 ** (np.log10(Mg_H * (1.00794 / 24.3050)) - AG_Mg)

    # Silicon, Sulphur & Calcium
    Si_H = sw_data.gas.element_mass_fractions.silicon.value / H
    Ca_Si = 0.0941736
    S_Si = 0.6054160
    ne_nH += Si_H * (1.00794 / 28.0855) * (14.0 / 1.0)
    ne_nH += (Si_H * Ca_Si) * (1.00794 / 40.078) * (20.0 / 1.0)
    ne_nH += (Si_H * S_Si) * (1.00794 / 32.065) * (16.0 / 1.0)
    ni_nH += Si_H * (1.00794 / 28.0855)
    ni_nH += (Si_H * Ca_Si) * (1.00794 / 40.078)
    ni_nH += (Si_H * S_Si) * (1.00794 / 32.065)
    mu += Si_H / (1.0 + 14.0)
    mu += (Si_H * Ca_Si) / (1.0 + 20.0)
    mu += (Si_H * S_Si) / (1.0 + 16.0)
    AG_Si = 7.55 - lN_H_AG
    AG_Ca = 6.36 - lN_H_AG
    AG_S = 7.21 - lN_H_AG
    Ca_H = 10.0 ** (np.log10((Ca_Si * Si_H) * (1.00794 / 40.078)) - AG_Ca)
    S_H = 10.0 ** (np.log10((S_Si * Si_H) * (1.00794 / 32.065)) - AG_S)
    Si_H = 10.0 ** (np.log10(Si_H * (1.00794 / 28.0855)) - AG_Si)

    # Iron
    Fe_H = sw_data.gas.element_mass_fractions.iron.value / H
    ne_nH += (Fe_H) * (1.00794 / 55.845) * (26.0 / 1.0)
    ni_nH += (Fe_H) * (1.00794 / 55.845)
    mu += (Fe_H) / (1.0 + 26.0)
    AG_Fe = 7.67 - lN_H_AG
    Fe_H = 10.0 ** (np.log10(Fe_H * (1.00794 / 55.845)) - AG_Fe)

    return ne_nH * (ne_nH / ((ne_nH + ni_nH) * mu)) ** 2
