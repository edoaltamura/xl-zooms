import os
import sys

from compare_density_profiles import density_profile_compare_plot
from contamination_mass import contamination_map, contamination_radial_histogram
from visualise_dm import dm_map_parent, dm_map_zoom
from compare_cumulative_mass import cumulative_mass_compare_plot

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from performance.wallclock_simulation_time import wallclock_simulation_time
from performance.number_of_steps_simulation_time import number_of_steps_simulation_time
from performance.particle_updates_step_cost import particle_updates_step_cost
from performance.wallclock_number_of_steps import wallclock_number_of_steps

dmo_repository = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo"

snap_filepath_parent = (
    "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
    "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
)

velociraptor_properties_parent = (
    "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
    "stf_swiftdm_3dfof_subhalo_0036/"
    "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
)


def dmo_diagnostics(run_name: str) -> None:
    snap_filepath_zoom = [
        f"/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/{run_name}/snapshots/{run_name}_0036.hdf5"
    ]

    velociraptor_properties_zoom = [
        f"/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/{run_name}/stf/{run_name}_0036/{run_name}_0036.properties"
    ]

    out_to_radius = (5, 'R200c')
    highres_radius = (6, 'R500c')

    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

    #####################################################################################################

    if not os.path.isdir(os.path.join(output_directory, run_name)):
        os.mkdir(os.path.join(output_directory, run_name))
    output_directory = os.path.join(output_directory, run_name)

    density_profile_compare_plot(
        run_name=run_name,
        snap_filepath_parent=snap_filepath_parent,
        velociraptor_properties_parent=velociraptor_properties_parent,
        snap_filepath_zoom=snap_filepath_zoom,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        output_directory=output_directory
    )

    cumulative_mass_compare_plot(
        run_name=run_name,
        snap_filepath_parent=snap_filepath_parent,
        velociraptor_properties_parent=velociraptor_properties_parent,
        snap_filepath_zoom=snap_filepath_zoom,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        output_directory=output_directory
    )

    for zoom_snap, zoom_vr in zip(
            snap_filepath_zoom,
            velociraptor_properties_zoom
    ):
        contamination_map(
            run_name=run_name,
            velociraptor_properties_zoom=zoom_vr,
            snap_filepath_zoom=zoom_snap,
            out_to_radius=out_to_radius,
            highres_radius=highres_radius,
            output_directory=output_directory,
        )

        contamination_radial_histogram(
            run_name=run_name,
            velociraptor_properties_zoom=zoom_vr,
            snap_filepath_zoom=zoom_snap,
            out_to_radius=out_to_radius,
            highres_radius=highres_radius,
            output_directory=output_directory,
        )

        # dm_map_parent(
        #     run_name=run_name,
        #     velociraptor_properties_parent=velociraptor_properties_parent,
        #     snap_filepath_parent=snap_filepath_parent,
        #     velociraptor_properties_zoom=zoom_vr,
        #     out_to_radius=out_to_radius,
        #     highres_radius=highres_radius,
        #     output_directory=output_directory,
        # )
        #
        # dm_map_zoom(
        #     run_name=run_name,
        #     snap_filepath_zoom=zoom_snap,
        #     velociraptor_properties_zoom=zoom_vr,
        #     out_to_radius=out_to_radius,
        #     highres_radius=highres_radius,
        #     output_directory=output_directory,
        # )

        wallclock_simulation_time(
            run_name=run_name,
            snap_filepath_zoom=zoom_snap,
            output_directory=output_directory,
        )

        number_of_steps_simulation_time(
            run_name=run_name,
            snap_filepath_zoom=zoom_snap,
            output_directory=output_directory,
        )

        particle_updates_step_cost(
            run_name=run_name,
            snap_filepath_zoom=zoom_snap,
            output_directory=output_directory,
        )

        wallclock_number_of_steps(
            run_name=run_name,
            snap_filepath_zoom=zoom_snap,
            output_directory=output_directory,
        )


if __name__ == '__main__':

    for run_id in os.listdir(dmo_repository):
        if (
                run_id.startswith('L0300N0564_VR') and
                run_id.endswith('-8res') and
                '1079' not in run_id and
                run_id not in os.listdir(os.path.join(output_directory, os.pardir))
        ):
            print(run_id)
            dmo_diagnostics(run_id)
