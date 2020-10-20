import numpy as np
import h5py as h5
from typing import Tuple


def find_nearest(array, value) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def find_object(
        vr_properties_catalog: str,
        sample_structType: int = 10,
        sample_mass_lower_lim: float = 1.e12,
        sample_M200c: float = None,
        sample_x: float = None,
        sample_y: float = None,
        sample_z: float = None,
        tolerance: float = 0.1,
) -> Tuple[int, float, float, float, float, float]:
    # Check that you have enough information for the queries
    arg_list = [sample_M200c, sample_x, sample_y, sample_z]
    number_valid_inputs = sum(1 for _ in filter(None.__ne__, arg_list))
    assert number_valid_inputs > 1, (
        f"Not enough valid inputs for the search. Need at least 2 non-None arguments, got {number_valid_inputs}."
    )

    # Read in halo properties
    with h5.File(vr_properties_catalog, 'r') as f:
        M200c = f['/Mass_200crit'][:] * 1.e10  # Msun units
        R200c = f['/R_200crit'][:]
        structType = f['/Structuretype'][:]
        xPotMin = f['/Xcminpot'][:]
        yPotMin = f['/Ycminpot'][:]
        zPotMin = f['/Zcminpot'][:]

    index = np.where((structType == sample_structType) & (M200c > sample_mass_lower_lim))[0]

    finder_result = dict()
    finder_result['name'] = []
    finder_result['value'] = []
    finder_result['index'] = []
    finder_result['error'] = []

    if sample_M200c is not None:
        assert sample_M200c > sample_mass_lower_lim, (
            "The mass of the object to search is below the lower limit for the mass cut. "
            f"`sample_mass_lower_lim` currently set to {sample_mass_lower_lim:.2f}."
        )
        _M200c_tuple = find_nearest(M200c[index], sample_M200c)
        print(_M200c_tuple)
        finder_result['name'].append('M200c')
        finder_result['value'].append(_M200c_tuple[0])
        finder_result['index'].append(_M200c_tuple[1])
        finder_result['error'].append(np.abs((_M200c_tuple[1] - sample_M200c) / sample_M200c))
        del _M200c_tuple
    if sample_x is not None:
        _x_tuple = find_nearest(xPotMin[index], sample_x)
        print(_x_tuple)
        finder_result['name'].append('x')
        finder_result['value'].append(_x_tuple[0])
        finder_result['index'].append(_x_tuple[1])
        finder_result['error'].append(np.abs((_x_tuple[1] - sample_x) / sample_x))
        del _x_tuple
    if sample_y is not None:
        _y_tuple = find_nearest(yPotMin[index], sample_y)
        print(_y_tuple)
        finder_result['name'].append('y')
        finder_result['value'].append(_y_tuple[0])
        finder_result['index'].append(_y_tuple[1])
        finder_result['error'].append(np.abs((_y_tuple[1] - sample_y) / sample_y))
        del _y_tuple
    if sample_z is not None:
        _z_tuple = find_nearest(zPotMin[index], sample_z)
        print(_z_tuple)
        finder_result['name'].append('z')
        finder_result['value'].append(_z_tuple[0])
        finder_result['index'].append(_z_tuple[1])
        finder_result['error'].append(np.abs((_z_tuple[1] - sample_z) / sample_z))
        del _z_tuple

    # Check that all queries return the same index
    assert len(set(finder_result['index'])) == 1, (
        f"Not all the queries returned the same output index for the VR catalogue. Found "
        f"{len(set(finder_result['index'])):d} different values ({set(finder_result['index'])}). "
        "Check that the inputs are correct, that their precision is sufficient and that you are "
        "providing enough data to match queries."
    )
    max_error = max(finder_result['error'])
    max_error_name = finder_result['name'][finder_result['error'].index(max(finder_result['error']))]
    max_error_index = finder_result['index'][finder_result['error'].index(max(finder_result['error']))]
    assert max(finder_result['error']) > tolerance, (
        f"At least one of the matched values deviates from the input more than {tolerance * 100:.2f}%. "
        "Large discrepancies can lead to the selection of the wrong object in the box.\n"
        f"Maximum error found >> name: {max_error_name:s} error: {max_error:.2f} index: {max_error_index:d}"
    )

    return tuple([
        finder_result['index'][0],
        M200c[index][finder_result['index'][0]],
        R200c[index][finder_result['index'][0]],
        xPotMin[index][finder_result['index'][0]],
        yPotMin[index][finder_result['index'][0]],
        zPotMin[index][finder_result['index'][0]]
    ])


if __name__ == "__main__":

    # TEST PARAMETERS ##########################################################

    test_M200c = np.asarray([1.487, 3.033, 6.959]) * 1e13
    test_R200c = np.asarray([0.519, 0.658, 0.868])
    test_x = np.asarray([134.688, 90.671, 71.962])
    test_y = np.asarray([169.921, 289.822, 69.291])
    test_z = np.asarray([289.233, 98.227, 240.338])

    haloPropFile = (
        "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/stf_swiftdm_3dfof_subhalo_0036/"
        "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
    )

    #############################################################################

    print("Testing with 4 inputs...")
    for i in range(3):
        try:
            results = find_object(
                vr_properties_catalog=haloPropFile,
                sample_M200c=test_M200c[i],
                sample_x=test_x[i],
                sample_y=test_y[i],
                sample_z=test_z[i],
            )
            print(results)
        except AssertionError as e:
            print(e)

    print("Testing with 3 inputs...")
    for i in range(3):
        try:
            results = find_object(
                vr_properties_catalog=haloPropFile,
                sample_x=test_x[i],
                sample_y=test_y[i],
                sample_z=test_z[i],
            )
            print(results)
        except AssertionError as e:
            print(e)

    print("Testing with 2 inputs...")
    for i in range(3):
        try:
            results = find_object(
                vr_properties_catalog=haloPropFile,
                sample_x=test_x[i],
                sample_y=test_y[i],
            )
            print(results)
        except AssertionError as e:
            print(e)

    print("Testing with 1 inputs...")
    for i in range(3):
        try:
            results = find_object(
                vr_properties_catalog=haloPropFile,
                sample_x=test_x[i],
            )
            print(results)
        except AssertionError as e:
            print(e)

    # with open("/cosma7/data/dp004/dc-alta2/xl-zooms/analysis/halo_selected_SK.txt", "w") as text_file:
    #     print("# Halo counter, M200c/1.e13 [Msun], r200c [Mpc], xPotMin [Mpc], yPotMin [Mpc], zPotMin [Mpc]",
    #           file=text_file)
    #
    # for i in range(3):
    #     print(f"Finding properties of halo {i:d}...")
    #     find_M200c = find_nearest(M200c, sample_M200c[i])
    #     find_R200c = find_nearest(R200c, sample_R200c[i])
    #     find_x = find_nearest(xPotMin, sample_x[i])
    #     find_y = find_nearest(yPotMin, sample_y[i])
    #     find_z = find_nearest(zPotMin, sample_z[i])
    #
    #     # Print to txt file
    #     print(i, find_M200c / 1.e13, find_R200c, find_x, find_y, find_z)
    #     with open("/cosma7/data/dp004/dc-alta2/xl-zooms/analysis/halo_selected_SK.txt", "a") as text_file:
    #         print(f"{i}, {find_M200c / 1.e13}, {find_R200c}, {find_x}, {find_y}, {find_z}", file=text_file)
