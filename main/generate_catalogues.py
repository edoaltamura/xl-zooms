import sys

sys.path.append("..")

from scaling_relations import *

print("[Analysis] Computing VRProperties...")
VRProperties().process_catalogue()
print("[Analysis] Computing GasFractions...")
GasFractions().process_catalogue()
print("[Analysis] Computing StarFractions...")
StarFractions().process_catalogue()
print("[Analysis] Computing MWTemperatures...")
MWTemperatures().process_catalogue()
print("[Analysis] Computing Relaxation...")
Relaxation().process_catalogue()
print("[Analysis] Computing XrayLuminosities...")
XrayLuminosities().process_catalogue()
print("[Analysis] Computing Entropies...")
Entropies().process_catalogue()
print("[Analysis] Computing SphericalOverdensities...")
SphericalOverdensities(density_contrast=1500.).process_catalogue()

