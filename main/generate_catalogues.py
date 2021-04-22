import sys

sys.path.append("..")

from scaling_relations import *

VRProperties().process_catalogue()
GasFractions().process_catalogue()
StarFractions().process_catalogue()
MWTemperatures().process_catalogue()
Relaxation().process_catalogue()
XrayLuminosities().process_catalogue()
Entropies().process_catalogue()
SphericalOverdensities(density_contrast=1500.).process_catalogue()

