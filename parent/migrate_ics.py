from os import mkdir, getcwd, chdir
from os.path import isdir, isfile, join
import subprocess
import yaml

from combine_ics import combine_ics as combine


def load_yaml(filename: str) -> dict:
    with open(filename, "r") as handle:
        return yaml.load(handle, Loader=yaml.Loader)


def create_new_parameter_file(parameter_file: dict, row: str, value) -> dict:
    new_parameter_file = parameter_file.copy()
    section, param = row.split(":")
    new_parameter_file[section][param] = value
    return new_parameter_file


def write_new_parameter_file(parameter_file: dict, filename: str) -> None:
    with open(filename, "w") as handle:
        yaml.dump(parameter_file, handle, default_flow_style=False)
    return


ic_gen_dir = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/ic_gen/run/ic_gen_submit_files"
swift_runs = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo"

runs = [
    "L0300N0564_VR1079_+1res",
    "L0300N0564_VR1079_-8res",
    "L0300N0564_VR121_-8res",
    "L0300N0564_VR1236_+1res",
    "L0300N0564_VR1236_-8res",
    "L0300N0564_VR130_-8res",
    "L0300N0564_VR139_+1res",
    "L0300N0564_VR139_-8res",
    "L0300N0564_VR155_+1res",
    "L0300N0564_VR155_-8res",
    "L0300N0564_VR18_+1res",
    "L0300N0564_VR1857_+1res",
    "L0300N0564_VR1857_-8res",
    "L0300N0564_VR187_+1res",
    "L0300N0564_VR187_-8res",
    "L0300N0564_VR18_-8res",
    "L0300N0564_VR2272_+1res",
    "L0300N0564_VR2272_-8res",
    "L0300N0564_VR23_+1res",
    "L0300N0564_VR23_-8res",
    "L0300N0564_VR2414_+1res",
    "L0300N0564_VR2414_-8res",
    "L0300N0564_VR2766_+1res",
    "L0300N0564_VR2766_-8res",
    "L0300N0564_VR2905_+1res",
    "L0300N0564_VR2905_-8res",
    "L0300N0564_VR2915_+1res",
    "L0300N0564_VR2915_-8res",
    "L0300N0564_VR3032_+1res",
    "L0300N0564_VR3032_-8res",
    "L0300N0564_VR340_-8res",
    "L0300N0564_VR36_+1res",
    "L0300N0564_VR36_-8res",
    "L0300N0564_VR37_+1res",
    "L0300N0564_VR37_-8res",
    "L0300N0564_VR470_-8res",
    "L0300N0564_VR485_+1res",
    "L0300N0564_VR485_-8res",
    "L0300N0564_VR55_+1res",
    "L0300N0564_VR55_-8res",
    "L0300N0564_VR645_+1res",
    "L0300N0564_VR645_-8res",
    "L0300N0564_VR666_+1res",
    "L0300N0564_VR666_-8res",
    "L0300N0564_VR680_+1res",
    "L0300N0564_VR680_-8res",
    "L0300N0564_VR775_+1res",
    "L0300N0564_VR775_-8res",
    "L0300N0564_VR801_+1res",
    "L0300N0564_VR801_-8res",
    "L0300N0564_VR813_+1res",
    "L0300N0564_VR813_-8res",
    "L0300N0564_VR918_+1res",
    "L0300N0564_VR918_-8res",
    "L0300N0564_VR93_+1res",
    "L0300N0564_VR93_-8res",
    "L0300N0564_VR93_+8res",
]

snap_outputs = ["# Redshift", "18.08", "15.28", "13.06", "11.26", "9.79", "8.57", "7.54", "6.67", "5.92", "5.28",
                "4.72", "4.24", "3.81", "3.43", "3.09", "2.79", "2.52", "2.28", "2.06", "1.86", "1.68", "1.51", "1.36",
                "1.21", "1.08", "0.96", "0.85", "0.74", "0.64", "0.55", "0.46", "0.37", "0.29", "0.21", "0.14", "0.07",
                "0.00"]

for run in runs:

    if "-8" in run:

        assert isdir(join(ic_gen_dir, run)), f"Run {run} not found in the IC_Gen directory."
        assert isdir(join(swift_runs, run)), f"Run {run} not found in the SWIFT directory."

        if not isdir(join(swift_runs, run, "ics")):
            mkdir(join(swift_runs, run, "ics"))

        if not isfile(join(swift_runs, run, "ics", run + ".hdf5")):
            assert isfile(join(ic_gen_dir, run, "ICs", run + ".0.hdf5")), \
                f"Run {run} does not have output files in the IC_Gen directory."

            print(f"Combining initial conditions: {run + '.x.hdf5'} >> {run + '.hdf5'}")
            combine(
                join(ic_gen_dir, run, "ICs", run + ".0.hdf5"),
                join(swift_runs, run, "ics", run + ".hdf5")
            )

        if not isdir(join(swift_runs, run, "config")):
            mkdir(join(swift_runs, run, "config"))

        # Handle the SWIFT parameter file
        assert isfile(join(swift_runs, run, "params.yml")), f"No SWIFT parameter file found for run {run}."

        print(f"Adapting SWIFT parameter file for {run}")
        param_file = load_yaml(join(swift_runs, run, "params.yml"))

        new_param_file = create_new_parameter_file(
            param_file,
            "InitialConditions:file_name",
            f"./ics/{run}.hdf5"
        )
        new_param_file = create_new_parameter_file(
            new_param_file,
            "Snapshots:output_list",
            f"./config/snap_redshifts.txt"
        )

        write_new_parameter_file(new_param_file, join(swift_runs, run, "params.yml"))

        # Create Snapshot:output_list file into ./config if not present
        if not isfile(join(swift_runs, run, "config", "snap_redshifts.txt")):
            with open(join(swift_runs, run, "config", "snap_redshifts.txt"), 'w') as f:
                for line in snap_outputs:
                    print(line, file=f)

        old_cwd = getcwd()
        chdir(join(swift_runs, run))
        print(f"Submit job to queue: {run}")
        subprocess.call(["sbatch", "submit"])
        chdir(old_cwd)
