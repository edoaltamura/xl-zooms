from compare_density_profiles import density_profile_compare_plot
from contamination_mass import contamination_map, contamination_radial_histogram
from visualise_dm import dm_map_parent, dm_map_zoom
from compare_cumulative_mass import cumulative_mass_compare_plot

run_name = "L0300N0564_VR93"

snap_filepath_parent = (
    "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
    "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
)

velociraptor_properties_parent = (
    "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
    "stf_swiftdm_3dfof_subhalo_0036/"
    "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
)

snap_filepath_zoom = [
    "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/snapshots/L0300N0564_VR93_0199.hdf5"
]

velociraptor_properties_zoom = [
    "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/properties"
]

out_to_radius = (5, 'R200c')
highres_radius = (6, 'R500c')

output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

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

    dm_map_parent(
        run_name=run_name,
        velociraptor_properties_parent=velociraptor_properties_parent,
        snap_filepath_parent=snap_filepath_parent,
        velociraptor_properties_zoom=zoom_vr,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )

    dm_map_zoom(
        run_name=run_name,
        snap_filepath_zoom=zoom_snap,
        velociraptor_properties_zoom=zoom_vr,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )
