# To integrate the zooms register in external codes, write the following import statement
#       from register import zooms_register
# To access the catalog file of the first zoom, use
#       zooms_register[0].catalog_file

from typing import List
import os


class Zoom:

    def __init__(
            self,
            run_name: str,
            snapshot_file: str,
            catalog_file: str,
            output_directory: str
    ) -> None:
        self.run_name = run_name
        self.snapshot_file = snapshot_file
        self.catalog_file = catalog_file
        self.output_directory = output_directory

    def __str__(self):
        return (
            "Zoom object:\n"
            f"\tName:                    {self.run_name}\n"
            f"\tSnapshot file:           {self.snapshot_file}\n"
            f"\tCatalog file:            {self.catalog_file}\n"
            f"\tOutput directory:        {self.output_directory}\n"
        )


def get_vr_number_from_name(name: str) -> int:
    start = 'VR'
    end = '_'
    start_position = name.find(start) + len(start)
    result = name[start_position:name.find(end, start_position)]

    return int(result)


def get_vr_numbers_unique() -> List[int]:
    vr_nums = [get_vr_number_from_name(name) for name in name_list]
    vr_nums = set(vr_nums)
    return list(vr_nums).sort()


def get_allpaths_from_last(path_z0: str) -> list:
    if path_z0.endswith('.hdf5'):
        # This is a snapshot
        snapdir = os.path.dirname(path_z0)
        allpaths = [os.path.join(snapdir, filepath) for filepath in os.listdir(snapdir) if filepath.endswith('.hdf5')]
        allpaths = [filepath for filepath in allpaths if os.path.isfile(filepath)]
        allpaths.sort(key=lambda x: int(x[-9:-5]))
        return allpaths

    else:
        # This is probably a VR output
        vrdir = os.path.dirname(os.path.dirname(path_z0))
        vr_out_section = path_z0.split('.')[-1]
        allsubpaths = [os.path.join(vrdir, catalog_dir) for catalog_dir in os.listdir(vrdir)]

        allpaths = []
        for subpath in allsubpaths:
            for filepath in os.listdir(subpath):
                filepath = os.path.join(subpath, filepath)
                if filepath.endswith(vr_out_section):
                    assert os.path.isfile(filepath)
                    allpaths.append(filepath)
                    break

        allpaths.sort(key=lambda x: int(x.rstrip('.' + vr_out_section)[-4:]))
        return allpaths


class ZoomList:

    def __init__(self, *args) -> None:
        args_iter = iter(args)
        arg_length_iter = len(next(args_iter))

        # Check that the input lists all have the same length
        assert all(len(arg_length) == arg_length_iter for arg_length in args_iter), (
            f"Input lists must have the same length. "
            f"Detected: {(f'{len(arg_length)}' for arg_length in args_iter)}"
        )

        # Parse list data into cluster objects
        obj_list = []
        for zoom_data in zip(*args):
            obj_list.append(Zoom(*zoom_data))

        self.obj_list = obj_list

    def get_list(self) -> List[Zoom]:
        return self.obj_list

    def __str__(self):
        message = ''
        for obj in self.obj_list:
            message += str(obj)
        return message


name_list = [
    "L0300N0564_VR121_-8res_Ref",
    "L0300N0564_VR1236_-8res_Ref",
    "L0300N0564_VR130_-8res_Ref",
    "L0300N0564_VR139_-8res_Ref",
    "L0300N0564_VR155_-8res_Ref",
    "L0300N0564_VR187_-8res_Ref",
    "L0300N0564_VR18_-8res_Ref",
    "L0300N0564_VR2272_-8res_Ref",
    "L0300N0564_VR23_-8res_Ref",
    "L0300N0564_VR2414_-8res_Ref",
    "L0300N0564_VR2766_-8res_Ref",
    "L0300N0564_VR2905_-8res_Ref",
    "L0300N0564_VR2915_-8res_Ref",
    "L0300N0564_VR3032_-8res_Ref",
    "L0300N0564_VR340_-8res_Ref",
    "L0300N0564_VR36_-8res_Ref",
    "L0300N0564_VR37_-8res_Ref",
    "L0300N0564_VR470_-8res_Ref",
    "L0300N0564_VR485_-8res_Ref",
    "L0300N0564_VR55_-8res_Ref",
    "L0300N0564_VR666_-8res_Ref",
    "L0300N0564_VR680_-8res_Ref",
    "L0300N0564_VR775_-8res_Ref",
    "L0300N0564_VR801_-8res_Ref",
    "L0300N0564_VR813_-8res_Ref",
    "L0300N0564_VR918_-8res_Ref",
    "L0300N0564_VR93_-8res_Ref",

    "L0300N0564_VR1236_-8res_MinimumDistance",
    "L0300N0564_VR139_-8res_MinimumDistance",
    "L0300N0564_VR187_-8res_MinimumDistance",
    "L0300N0564_VR18_-8res_MinimumDistance",
    "L0300N0564_VR2414_-8res_MinimumDistance",
    "L0300N0564_VR2905_-8res_MinimumDistance",
    "L0300N0564_VR3032_-8res_MinimumDistance",
    "L0300N0564_VR470_-8res_MinimumDistance",
    "L0300N0564_VR55_-8res_MinimumDistance",
    "L0300N0564_VR666_-8res_MinimumDistance",
    "L0300N0564_VR813_-8res_MinimumDistance",
    "L0300N0564_VR93_-8res_MinimumDistance",

    "L0300N0564_VR1236_-8res_Isotropic",
    "L0300N0564_VR139_-8res_Isotropic",
    "L0300N0564_VR187_-8res_Isotropic",
    "L0300N0564_VR18_-8res_Isotropic",
    "L0300N0564_VR2414_-8res_Isotropic",
    "L0300N0564_VR2905_-8res_Isotropic",
    "L0300N0564_VR3032_-8res_Isotropic",
    "L0300N0564_VR470_-8res_Isotropic",
    "L0300N0564_VR55_-8res_Isotropic",
    "L0300N0564_VR666_-8res_Isotropic",
    "L0300N0564_VR813_-8res_Isotropic",
    "L0300N0564_VR93_-8res_Isotropic",

    "L0300N0564_VR3032_+1res_Isotropic",
    "L0300N0564_VR2905_+1res_Isotropic",
    "L0300N0564_VR2414_+1res_Isotropic",
    "L0300N0564_VR1236_+1res_Isotropic",
    "L0300N0564_VR813_+1res_Isotropic",
    "L0300N0564_VR666_+1res_Isotropic",

    "L0300N0564_VR3032_+1res_MinimumDistance",
    "L0300N0564_VR2905_+1res_MinimumDistance",
    "L0300N0564_VR2414_+1res_MinimumDistance",
    "L0300N0564_VR1236_+1res_MinimumDistance",
    "L0300N0564_VR813_+1res_MinimumDistance",
    "L0300N0564_VR666_+1res_MinimumDistance",
]

snapshot_filenames = [
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR121_-8res_Ref/snapshots/L0300N0564_VR121_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_Ref/snapshots/L0300N0564_VR1236_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR130_-8res_Ref/snapshots/L0300N0564_VR130_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_Ref/snapshots/L0300N0564_VR139_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR155_-8res_Ref/snapshots/L0300N0564_VR155_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_Ref/snapshots/L0300N0564_VR187_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_Ref/snapshots/L0300N0564_VR18_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2272_-8res_Ref/snapshots/L0300N0564_VR2272_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR23_-8res_Ref/snapshots/L0300N0564_VR23_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_Ref/snapshots/L0300N0564_VR2414_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2766_-8res_Ref/snapshots/L0300N0564_VR2766_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_Ref/snapshots/L0300N0564_VR2905_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2915_-8res_Ref/snapshots/L0300N0564_VR2915_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_Ref/snapshots/L0300N0564_VR3032_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR340_-8res_Ref/snapshots/L0300N0564_VR340_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR36_-8res_Ref/snapshots/L0300N0564_VR36_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR37_-8res_Ref/snapshots/L0300N0564_VR37_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_Ref/snapshots/L0300N0564_VR470_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR485_-8res_Ref/snapshots/L0300N0564_VR485_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_Ref/snapshots/L0300N0564_VR55_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_Ref/snapshots/L0300N0564_VR666_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR680_-8res_Ref/snapshots/L0300N0564_VR680_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR775_-8res_Ref/snapshots/L0300N0564_VR775_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR801_-8res_Ref/snapshots/L0300N0564_VR801_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_Ref/snapshots/L0300N0564_VR813_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR918_-8res_Ref/snapshots/L0300N0564_VR918_-8res_Ref_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_Ref/snapshots/L0300N0564_VR93_-8res_Ref_2749.hdf5",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_MinimumDistance/snapshots/L0300N0564_VR1236_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_MinimumDistance/snapshots/L0300N0564_VR139_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_MinimumDistance/snapshots/L0300N0564_VR187_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance/snapshots/L0300N0564_VR18_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_MinimumDistance/snapshots/L0300N0564_VR2414_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_MinimumDistance/snapshots/L0300N0564_VR2905_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_MinimumDistance/snapshots/L0300N0564_VR3032_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_MinimumDistance/snapshots/L0300N0564_VR470_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_MinimumDistance/snapshots/L0300N0564_VR55_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_MinimumDistance/snapshots/L0300N0564_VR666_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_MinimumDistance/snapshots/L0300N0564_VR813_-8res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_MinimumDistance/snapshots/L0300N0564_VR93_-8res_MinimumDistance_2749.hdf5",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_Isotropic/snapshots/L0300N0564_VR1236_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_Isotropic/snapshots/L0300N0564_VR139_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_Isotropic/snapshots/L0300N0564_VR187_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_Isotropic/snapshots/L0300N0564_VR18_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_Isotropic/snapshots/L0300N0564_VR2414_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_Isotropic/snapshots/L0300N0564_VR2905_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic/snapshots/L0300N0564_VR3032_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_Isotropic/snapshots/L0300N0564_VR470_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_Isotropic/snapshots/L0300N0564_VR55_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_Isotropic/snapshots/L0300N0564_VR666_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_Isotropic/snapshots/L0300N0564_VR813_-8res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_Isotropic/snapshots/L0300N0564_VR93_-8res_Isotropic_2749.hdf5",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_+1res_Isotropic/snapshots/L0300N0564_VR3032_+1res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_+1res_Isotropic/snapshots/L0300N0564_VR2905_+1res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_Isotropic/snapshots/L0300N0564_VR2414_+1res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_+1res_Isotropic/snapshots/L0300N0564_VR1236_+1res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_Isotropic/snapshots/L0300N0564_VR813_+1res_Isotropic_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_+1res_Isotropic/snapshots/L0300N0564_VR666_+1res_Isotropic_2749.hdf5",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_+1res_MinimumDistance/snapshots/L0300N0564_VR3032_+1res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_+1res_MinimumDistance/snapshots/L0300N0564_VR2905_+1res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_MinimumDistance/snapshots/L0300N0564_VR2414_+1res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_+1res_MinimumDistance/snapshots/L0300N0564_VR1236_+1res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/snapshots/L0300N0564_VR813_+1res_MinimumDistance_2749.hdf5",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_+1res_MinimumDistance/snapshots/L0300N0564_VR666_+1res_MinimumDistance_2749.hdf5",

]

catalogue_filenames = [
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR121_-8res_Ref/stf/L0300N0564_VR121_-8res_Ref_2749/L0300N0564_VR121_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_Ref/stf/L0300N0564_VR1236_-8res_Ref_2749/L0300N0564_VR1236_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR130_-8res_Ref/stf/L0300N0564_VR130_-8res_Ref_2749/L0300N0564_VR130_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_Ref/stf/L0300N0564_VR139_-8res_Ref_2749/L0300N0564_VR139_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR155_-8res_Ref/stf/L0300N0564_VR155_-8res_Ref_2749/L0300N0564_VR155_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_Ref/stf/L0300N0564_VR187_-8res_Ref_2749/L0300N0564_VR187_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_Ref/stf/L0300N0564_VR18_-8res_Ref_2749/L0300N0564_VR18_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2272_-8res_Ref/stf/L0300N0564_VR2272_-8res_Ref_2749/L0300N0564_VR2272_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR23_-8res_Ref/stf/L0300N0564_VR23_-8res_Ref_2749/L0300N0564_VR23_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_Ref/stf/L0300N0564_VR2414_-8res_Ref_2749/L0300N0564_VR2414_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2766_-8res_Ref/stf/L0300N0564_VR2766_-8res_Ref_2749/L0300N0564_VR2766_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_Ref/stf/L0300N0564_VR2905_-8res_Ref_2749/L0300N0564_VR2905_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2915_-8res_Ref/stf/L0300N0564_VR2915_-8res_Ref_2749/L0300N0564_VR2915_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_Ref/stf/L0300N0564_VR3032_-8res_Ref_2749/L0300N0564_VR3032_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR340_-8res_Ref/stf/L0300N0564_VR340_-8res_Ref_2749/L0300N0564_VR340_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR36_-8res_Ref/stf/L0300N0564_VR36_-8res_Ref_2749/L0300N0564_VR36_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR37_-8res_Ref/stf/L0300N0564_VR37_-8res_Ref_2749/L0300N0564_VR37_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_Ref/stf/L0300N0564_VR470_-8res_Ref_2749/L0300N0564_VR470_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR485_-8res_Ref/stf/L0300N0564_VR485_-8res_Ref_2749/L0300N0564_VR485_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_Ref/stf/L0300N0564_VR55_-8res_Ref_2749/L0300N0564_VR55_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_Ref/stf/L0300N0564_VR666_-8res_Ref_2749/L0300N0564_VR666_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR680_-8res_Ref/stf/L0300N0564_VR680_-8res_Ref_2749/L0300N0564_VR680_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR775_-8res_Ref/stf/L0300N0564_VR775_-8res_Ref_2749/L0300N0564_VR775_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR801_-8res_Ref/stf/L0300N0564_VR801_-8res_Ref_2749/L0300N0564_VR801_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_Ref/stf/L0300N0564_VR813_-8res_Ref_2749/L0300N0564_VR813_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR918_-8res_Ref/stf/L0300N0564_VR918_-8res_Ref_2749/L0300N0564_VR918_-8res_Ref_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_Ref/stf/L0300N0564_VR93_-8res_Ref_2749/L0300N0564_VR93_-8res_Ref_2749.properties",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_MinimumDistance/stf/L0300N0564_VR1236_-8res_MinimumDistance_2749/L0300N0564_VR1236_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_MinimumDistance/stf/L0300N0564_VR139_-8res_MinimumDistance_2749/L0300N0564_VR139_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_MinimumDistance/stf/L0300N0564_VR187_-8res_MinimumDistance_2749/L0300N0564_VR187_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance/stf/L0300N0564_VR18_-8res_MinimumDistance_2749/L0300N0564_VR18_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_MinimumDistance/stf/L0300N0564_VR2414_-8res_MinimumDistance_2749/L0300N0564_VR2414_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_MinimumDistance/stf/L0300N0564_VR2905_-8res_MinimumDistance_2749/L0300N0564_VR2905_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_MinimumDistance/stf/L0300N0564_VR3032_-8res_MinimumDistance_2749/L0300N0564_VR3032_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_MinimumDistance/stf/L0300N0564_VR470_-8res_MinimumDistance_2749/L0300N0564_VR470_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_MinimumDistance/stf/L0300N0564_VR55_-8res_MinimumDistance_2749/L0300N0564_VR55_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_MinimumDistance/stf/L0300N0564_VR666_-8res_MinimumDistance_2749/L0300N0564_VR666_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_MinimumDistance/stf/L0300N0564_VR813_-8res_MinimumDistance_2749/L0300N0564_VR813_-8res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_MinimumDistance/stf/L0300N0564_VR93_-8res_MinimumDistance_2749/L0300N0564_VR93_-8res_MinimumDistance_2749.properties",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_-8res_Isotropic/stf/L0300N0564_VR1236_-8res_Isotropic_2749/L0300N0564_VR1236_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR139_-8res_Isotropic/stf/L0300N0564_VR139_-8res_Isotropic_2749/L0300N0564_VR139_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR187_-8res_Isotropic/stf/L0300N0564_VR187_-8res_Isotropic_2749/L0300N0564_VR187_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR18_-8res_Isotropic/stf/L0300N0564_VR18_-8res_Isotropic_2749/L0300N0564_VR18_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_-8res_Isotropic/stf/L0300N0564_VR2414_-8res_Isotropic_2749/L0300N0564_VR2414_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_-8res_Isotropic/stf/L0300N0564_VR2905_-8res_Isotropic_2749/L0300N0564_VR2905_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic/stf/L0300N0564_VR3032_-8res_Isotropic_2749/L0300N0564_VR3032_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR470_-8res_Isotropic/stf/L0300N0564_VR470_-8res_Isotropic_2749/L0300N0564_VR470_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR55_-8res_Isotropic/stf/L0300N0564_VR55_-8res_Isotropic_2749/L0300N0564_VR55_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_-8res_Isotropic/stf/L0300N0564_VR666_-8res_Isotropic_2749/L0300N0564_VR666_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_-8res_Isotropic/stf/L0300N0564_VR813_-8res_Isotropic_2749/L0300N0564_VR813_-8res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR93_-8res_Isotropic/stf/L0300N0564_VR93_-8res_Isotropic_2749/L0300N0564_VR93_-8res_Isotropic_2749.properties",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_+1res_Isotropic/stf/L0300N0564_VR3032_+1res_Isotropic_2749/L0300N0564_VR3032_+1res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_+1res_Isotropic/stf/L0300N0564_VR2905_+1res_Isotropic_2749/L0300N0564_VR2905_+1res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_Isotropic/stf/L0300N0564_VR2414_+1res_Isotropic_2749/L0300N0564_VR2414_+1res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_+1res_Isotropic/stf/L0300N0564_VR1236_+1res_Isotropic_2749/L0300N0564_VR1236_+1res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_Isotropic/stf/L0300N0564_VR813_+1res_Isotropic_2749/L0300N0564_VR813_+1res_Isotropic_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_+1res_Isotropic/stf/L0300N0564_VR666_+1res_Isotropic_2749/L0300N0564_VR666_+1res_Isotropic_2749.properties",

    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR3032_+1res_MinimumDistance/stf/L0300N0564_VR3032_+1res_MinimumDistance_2749/L0300N0564_VR3032_+1res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2905_+1res_MinimumDistance/stf/L0300N0564_VR2905_+1res_MinimumDistance_2749/L0300N0564_VR2905_+1res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_MinimumDistance/stf/L0300N0564_VR2414_+1res_MinimumDistance_2749/L0300N0564_VR2414_+1res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR1236_+1res_MinimumDistance/stf/L0300N0564_VR1236_+1res_MinimumDistance_2749/L0300N0564_VR1236_+1res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/stf/L0300N0564_VR813_+1res_MinimumDistance_2749/L0300N0564_VR813_+1res_MinimumDistance_2749.properties",
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR666_+1res_MinimumDistance/stf/L0300N0564_VR666_+1res_MinimumDistance_2749/L0300N0564_VR666_+1res_MinimumDistance_2749.properties",

]

output_dir = ["/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"] * len(catalogue_filenames)

Tcut_halogas = 1.e5  # Hot gas temperature threshold in K

zooms_register = ZoomList(
    name_list,
    snapshot_filenames,
    catalogue_filenames,
    output_dir,
).obj_list

vr_numbers = get_vr_numbers_unique()

if __name__ == "__main__":
    for i in zooms_register:
        print(i)
