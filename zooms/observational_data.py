import numpy as np
from astropy import cosmology
from unyt import Solar_Mass


class Observations:

    def __init__(self, cosmo_model: str = "Planck18"):

        for model_name in dir(cosmology):
            if cosmo_model.lower() in model_name.lower():
                print(f"Using the {model_name} cosmology")
                self.cosmo_model = getattr(cosmology, model_name)


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
    notes = "Data in h_70 units."

    M500_Bud = np.array([10 ** 13.7, 10 ** 15]) * Solar_Mass
    Mstar500_Bud = 10. ** (0.89 * np.log10(M500_Bud / 3.e14) + 12.44) * Solar_Mass

    # Mstar−M500 relation
    # a = 0.89 ± 0.14 and b = 12.44 ± 0.03
    #
    # $\log \left(\frac{M_{\text {star }}}{\mathrm{M}_{\odot}}\right)=a \log \left(\frac{M_{500}}
    # {3 \times 10^{14} \mathrm{M}_{\odot}}\right)+b$

    # star−M500 relation
    # alpha = −0.11 ± 0.14 and beta = −2.04 ± 0.03
    #
    # $\log f_{\text {star }}=\alpha \log \left(\frac{M_{500}}{3 \times 10^{14} \mathrm{M}_{\odot}}\right)+\beta$

    def __init__(self, *args, **kwargs):
        super(Budzynski14, self).__init__(*args, **kwargs)

        h70_Bud = 0.70 / self.cosmo_model.h
        self.M500 = self.M500_Bud * h70_Bud
        self.Mstar500 = self.Mstar500_Bud * (h70_Bud ** 2.5)


class Gonzalez13(Observations):
    pass


class Barnes17(Observations):
    pass


bud14 = Budzynski14()
print(bud14.Mstar500, bud14.M500)
