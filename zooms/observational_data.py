import numpy as np


class Observations:

    def __init__(self):
        pass


class Sun09(Observations):
    paper_name = "Sun et al. (2009)"

    M500_Sun = np.array(
        [3.18, 4.85, 3.90, 1.48, 4.85, 5.28, 8.49, 10.3, 2.0, 7.9, 5.6, 12.9, 8.0, 14.1, 3.22, 14.9, 13.4, 6.9, 8.95,
         8.8, 8.3, 9.7, 7.9]
    )
    f500_Sun = np.array(
        [0.097, 0.086, 0.068, 0.049, 0.069, 0.060, 0.076, 0.081, 0.108, 0.086, 0.056, 0.076, 0.075, 0.114, 0.074, 0.088,
         0.094, 0.094, 0.078, 0.099, 0.065, 0.090, 0.093]
    )
    Mgas500_Sun = M500_Sun * f500_Sun

    # Convert units to h_70
    h70_Sun = 73. / 70.
    M500_Sun *= h70_Sun
    Mgas500_Sun *= (h70_Sun ** 2.5)


class Lovisari15(Observations):
    paper_name = "Lovisari et al. (2015)"
    notes = "Data in h_70 units already."

    # Lovisari et al. 2015 (in h_70 units already)
    M500_Lov = np.array(
        [2.07, 4.67, 2.39, 2.22, 2.95, 2.83, 3.31, 3.53, 3.49, 3.35, 14.4, 2.34, 4.78, 8.59, 9.51, 6.96, 10.8, 4.37,
         8.00, 12.1]
    )
    Mgas500_Lov = np.array(
        [0.169, 0.353, 0.201, 0.171, 0.135, 0.272, 0.171, 0.271, 0.306, 0.247, 1.15, 0.169, 0.379, 0.634, 0.906, 0.534,
         0.650, 0.194, 0.627, 0.817]
    )
