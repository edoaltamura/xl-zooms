# To integrate the zooms register in external codes, write the following import statement
#       from register import zooms_register
# To access the catalog file of the first zoom, use
#       zooms_register[0].catalog_file
import io
import os
import h5py
import zipfile
from typing import List
from tqdm import tqdm
import psutil

SILENT_PROGRESSBAR = False
total_memory = psutil.virtual_memory().total


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


def get_allpaths_from_last(path_z0: str, z_min: float = 0., z_max: float = 5.) -> list:
    if path_z0.endswith('.hdf5'):
        # This is a snapshot
        snapdir = os.path.dirname(path_z0)
        allpaths = [os.path.join(snapdir, filepath) for filepath in os.listdir(snapdir) if filepath.endswith('.hdf5')]
        allpaths = [filepath for filepath in allpaths if os.path.isfile(filepath)]
        allpaths.sort(key=lambda x: int(x[-9:-5]))
        # Filter redshifts
        for path in tqdm(allpaths.copy(), desc=f"Fetching snapshots", disable=SILENT_PROGRESSBAR):
            with h5py.File(path, 'r') as f:
                z = f['Header'].attrs['Redshift'][0]
            if z > z_max or z < z_min:
                allpaths.remove(path)
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
        # Filter redshifts
        for path in tqdm(allpaths.copy(), desc=f"Fetching snap catalogues", disable=SILENT_PROGRESSBAR):
            with h5py.File(path, 'r') as file:
                scale_factor = float(file['/SimulationInfo'].attrs['ScaleFactor'])
                z = 1 / scale_factor - 1
            if z > z_max or z < z_min:
                allpaths.remove(path)
        return allpaths


def get_snip_handles(path_z0: str, z_min: float = 0., z_max: float = 5.):
    snip_path = os.path.dirname(path_z0)
    snip_handles = []

    with zipfile.ZipFile(os.path.join(snip_path, 'snipshots.zip'), 'r') as archive:
        all_snips = archive.namelist()
        all_snips.sort(key=lambda x: int(x[-9:-5]))
        for snip_name in tqdm(all_snips, desc=f"Fetching snipshots", disable=SILENT_PROGRESSBAR):
            snip_handle = io.BytesIO(archive.open(snip_name).read())
            with h5py.File(snip_handle, 'r') as f:
                z = f['Header'].attrs['Redshift'][0]
            # Filter redshifts
            if z_min < z < z_max:
                snip_handles.append(snip_handle)

    return snip_handles


def dump_memory_usage() -> None:
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss  # in bytes
    print((
        f"[Resources] Memory usage: {memory / 1024 / 1024:.2f} MB "
        f"({memory / total_memory * 100:.2f}%)"
    ))


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

cosma_repositories = [
    "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro",
    "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro",
    "/snap7/scratch/dp004/dc-alta2/xl-zooms/hydro",
]

name_list = []
snapshot_filenames = []
catalogue_filenames = []

for repo in cosma_repositories:
    for run_dir in os.listdir(repo):
        run_path = os.path.join(repo, run_dir)

        if (
                run_dir.startswith('L0300N0564') and
                os.path.isdir(run_path) and
                os.path.isdir(os.path.join(run_path, 'snapshots')) and
                os.path.isdir(os.path.join(run_path, 'stf'))
        ):
            name_list.append(run_dir)

            snap_files = os.listdir(os.path.join(run_path, 'snapshots'))
            snap_files = [file_name for file_name in snap_files if file_name.endswith('.hdf5')]
            sorted(snap_files, key=lambda x: int(x.split('_')[-1][:4]))
            snap_z0 = snap_files[-1]
            snap_z0_path = os.path.join(run_path, 'snapshots', snap_z0)
            assert os.path.isfile(snap_z0_path)
            snapshot_filenames.append(snap_z0_path)

            catalogue_filenames.append(
                os.path.join(
                    run_path,
                    'stf',
                    os.path.splitext(snap_z0)[0],
                    f"{os.path.splitext(snap_z0)[0]}.properties"
                )
            )
        elif (
                run_dir.startswith('L0300N0564') and
                os.path.isdir(run_path) and (
                        ~os.path.isdir(os.path.join(run_path, 'snapshots')) or
                        ~os.path.isdir(os.path.join(run_path, 'stf'))
                )
        ):
            print((
                f"The simulation {run_dir} was found with directory set-up, "
                "but was missing the snapshots or stf sub-directory. It was "
                "likely not launched and was not appended in the master register."
            ))

output_dir = ["/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"] * len(name_list)

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
