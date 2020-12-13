import unyt

eagle_dm_mass = unyt.unyt_quantity(1.e6, unyt.Solar_Mass)
eagle_gas_mass = unyt.unyt_quantity(1.e6, unyt.Solar_Mass)

register = [
    {
        'Name': "L0300N0564_VR121_-8res",
        'Resolution': '-8res',
        'DM particle mass': eagle_dm_mass * 8,
        'Initial gas mass': eagle_gas_mass * 8,

    }
]
