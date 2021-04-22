def calculate_means(data):
    center = data.metadata.boxsize / 2.

    r_gas = np.sqrt(np.sum((data.gas.coordinates - center) ** 2, axis=1))
    bins = radial_bins.to(r_gas.units)

    masses = data.gas.masses.value
    temp = data.gas.temperatures

    mass_avg_T = normalized_mean(r_gas, temp, masses, bins) * K

    density = data.gas.densities
    density.convert_to_units(0.759 * mh / (cm ** 3))

    mass_avg_density = normalized_mean(r_gas, density.value, masses, bins) * data.gas.densities.units
    mass_avg_density2 = normalized_mean(r_gas, density.value * density.value, masses,
                                        bins) * data.gas.densities.units ** 2

    entropy = unyt.kb * temp / ((density / unyt.mh) ** (2. / 3.))
    entropy.convert_to_units(unyt.keV * cm ** 2)
    mass_avg_entropy = normalized_mean(r_gas, entropy.value, masses, bins) * entropy.units

    internal_energy = data.gas.internal_energies
    pressure = (5. / 3. - 1) * internal_energy * density
    pressure.convert_to_units(unyt.g / cm / unyt.s ** 2)
    mass_avg_pressure = normalized_mean(r_gas, pressure, masses, bins) * pressure.units

    volumes = bin_volumes(bins)  # * r_gas.units**3

    mass_in_bins, _, _ = stat.binned_statistic(x=r_gas, values=masses, statistic="sum", bins=bins)
    mass_in_bins *= data.gas.masses.units

    V1 = 4 * np.pi / 3 * bins ** 3
    volume2 = V1[1:] - V1[:-1]
    density_mass_shells = mass_in_bins / volumes
    density_mass_shells.convert_to_units(0.759 * mh / (cm ** 3))

    volume_proxy = masses / density
    volume_proxy = volume_proxy.value

    volume_avg_T = normalized_mean(r_gas, temp, volume_proxy, bins) * K

    volume_avg_density = normalized_mean(r_gas, density.value, volume_proxy, bins) * data.gas.densities.units

    volume_avg_density2 = normalized_mean(r_gas, density.value * density.value, volume_proxy,
                                          bins) * data.gas.densities.units ** 2

    volume_avg_entropy = normalized_mean(r_gas, entropy.value, volume_proxy, bins) * entropy.units
    volume_avg_pressure = normalized_mean(r_gas, pressure, volume_proxy, bins) * pressure.units

    return mass_avg_T, mass_avg_density, mass_avg_density2, mass_avg_entropy, volume_avg_T, volume_avg_density, volume_avg_density2, volume_avg_entropy, mass_avg_pressure, volume_avg_pressure


def calculate_log_means(data):
    center = data.metadata.boxsize / 2.

    r_gas = np.sqrt(np.sum((data.gas.coordinates - center) ** 2, axis=1))
    bins = radial_bins.to(r_gas.units)

    masses = data.gas.masses
    masses.convert_to_units(Msun)
    log_temp = np.log10(data.gas.temperatures)
    temp = data.gas.temperatures

    mass_avg_T = 10 ** normalized_mean(r_gas, log_temp, masses.value, bins) * K

    density = data.gas.densities
    density.convert_to_units(0.759 * mh / (cm ** 3))
    log_density = np.log10(data.gas.densities.value)

    mass_avg_density = 10 ** normalized_mean(r_gas, log_density, masses.value, bins) * data.gas.densities.units
    mass_avg_density2 = 10 ** normalized_mean(r_gas, log_density * log_density, masses.value,
                                              bins) * data.gas.densities.units ** 2

    entropy = unyt.kb * temp / ((density / unyt.mh) ** (2. / 3.))
    entropy.convert_to_units(unyt.keV * cm ** 2)

    log_entropy = np.log10(entropy.value)
    mass_avg_entropy = 10 ** normalized_mean(r_gas, log_entropy, masses.value, bins) * entropy.units

    internal_energy = data.gas.internal_energies
    pressure = (5. / 3. - 1) * internal_energy * density
    pressure.convert_to_units(unyt.g / cm / unyt.s ** 2)
    log_pressure = np.log10(pressure)
    mass_avg_pressure = 10 ** normalized_mean(r_gas, log_pressure, masses.value, bins) * pressure.units

    volume_proxy = masses / density
    volume_proxy = volume_proxy.value

    volume_avg_T = 10 ** normalized_mean(r_gas, log_temp, volume_proxy, bins) * K

    volume_avg_density = 10 ** normalized_mean(r_gas, log_density, volume_proxy, bins) * data.gas.densities.units
    volume_avg_density2 = 10 ** normalized_mean(r_gas, log_density * log_density, volume_proxy,
                                                bins) * data.gas.densities.units ** 2

    volume_avg_entropy = 10 ** normalized_mean(r_gas, log_entropy, volume_proxy, bins) * entropy.units
    volume_avg_pressure = 10 ** normalized_mean(r_gas, log_pressure, volume_proxy, bins) * pressure.units

    return mass_avg_T, mass_avg_density, mass_avg_density2, mass_avg_entropy, volume_avg_T, volume_avg_density, volume_avg_density2, volume_avg_entropy, mass_avg_pressure, volume_avg_pressure


def calculate_Xray_weighted_means(data):
    center = data.metadata.boxsize / 2.

    r_gas = np.sqrt(np.sum((data.gas.coordinates - center) ** 2, axis=1))
    bins = radial_bins.to(r_gas.units)

    print(data.gas.Xray_erg_per_s)
    Xray_lum = data.gas.Xray_erg_per_s.value
    temp = data.gas.temperatures

    avg_T = normalized_mean(r_gas, temp, Xray_lum, bins) * K

    density = data.gas.densities
    density.convert_to_units(0.759 * mh / (cm ** 3))

    avg_density = normalized_mean(r_gas, density.value, Xray_lum, bins) * data.gas.densities.units
    avg_density2 = normalized_mean(r_gas, density.value * density.value, Xray_lum, bins) * data.gas.densities.units ** 2

    entropy = unyt.kb * temp / ((density / unyt.mh) ** (2. / 3.))
    entropy.convert_to_units(unyt.keV * cm ** 2)
    avg_entropy = normalized_mean(r_gas, entropy.value, Xray_lum, bins) * entropy.units

    internal_energy = data.gas.internal_energies
    pressure = (5. / 3. - 1) * internal_energy * density
    pressure.convert_to_units(unyt.g / cm / unyt.s ** 2)
    avg_pressure = normalized_mean(r_gas, pressure, Xray_lum, bins) * pressure.units

    Z = data.gas.metal_mass_fractions.value / 0.0133714  # in solar
    v2 = data.gas.velocity_dispersions
    sound_velocity2 = 5. / 3. * kb * data.gas.temperatures / (1.3 * mh)
    ratio_v_sound = (v2 / sound_velocity2) ** .5

    avg_Z = normalized_mean(r_gas, Z, Xray_lum, bins)

    avg_mag_number = normalized_mean(r_gas, ratio_v_sound, Xray_lum, bins)

    total_Xray, _, _ = stat.binned_statistic(
        x=r_gas, values=Xray_lum, statistic="sum", bins=bins
    )  # * data.gas.Xray_erg_per_s.units

    volumes = bin_volumes(bins)  # * r_gas.units**3
    total_Xray = total_Xray * data.gas.Xray_erg_per_s.units / volumes

    r_xy = np.sqrt((data.gas.coordinates[:, 0] - center[0]) ** 2 + (data.gas.coordinates[:, 1] - center[1]) ** 2)

    total_Xray2, _, _ = stat.binned_statistic(
        x=r_xy, values=Xray_lum, statistic="sum", bins=bins
    )  # * data.gas.Xray_erg_per_s.units

    areas = bin_areas(bins)
    total_Xray2 = total_Xray2 * data.gas.Xray_erg_per_s.units / areas

    # mass_avg_Xray = normalized_mean(r_gas, Xray_lum, masses, bins) * data.gas.Xray_erg_per_s.units
    # volume_avg_Xray = normalized_mean(r_gas, Xray_lum, volume_proxy, bins) * data.gas.Xray_erg_per_s.units

    return avg_T, avg_density, avg_density2, avg_entropy, avg_pressure, total_Xray, total_Xray2, avg_Z, avg_mag_number  # mass_avg_Xray, volume_avg_Xray


def calculate_cumulative_mass(data, R500):
    masses = data.gas.masses
    masses.convert_to_units(unyt.Msun)
    center = data.metadata.boxsize / 2.

    r_gas = np.sqrt(np.sum((data.gas.coordinates - center) ** 2, axis=1))
    r_gas.convert_to_units(unyt.Mpc)

    hist, bin_edges = np.histogram(np.log10(r_gas) - np.log10(R500), bins=100, range=[-3, 1], weights=masses)
    return (bin_edges[1:] + bin_edges[:-1]) / 2., np.cumsum(hist)


def calculate_means_others(data):
    center = data.metadata.boxsize / 2.

    r_gas = np.sqrt(np.sum((data.gas.coordinates - center) ** 2, axis=1))
    bins = radial_bins.to(r_gas.units)

    masses = data.gas.masses.value
    density = data.gas.densities
    volume_proxy = masses / density
    volume_proxy = volume_proxy.value

    Z = data.gas.metal_mass_fractions.value / 0.0133714  # in solar
    v2 = data.gas.velocity_dispersions
    sound_velocity2 = 5. / 3. * kb * data.gas.temperatures / (1.3 * mh)
    ratio_v_sound = (v2 / sound_velocity2) ** .5

    mass_avg_Z = normalized_mean(r_gas, Z, masses, bins)
    volume_avg_Z = normalized_mean(r_gas, Z, volume_proxy, bins)

    mass_avg_mag_number = normalized_mean(r_gas, ratio_v_sound, masses, bins)
    volume_avg_mag_number = normalized_mean(r_gas, ratio_v_sound, volume_proxy, bins)

    return mass_avg_Z, volume_avg_Z, mass_avg_mag_number, volume_avg_mag_number


def calculate_mean_cooling_times(data):
    tff = np.sqrt(3 * np.pi / (32 * G * data.gas.densities))

    data_cooling = get_cooling_rates()
    data_heating = get_heating_rates()

    cooling_rates = np.log10(np.power(10., data_cooling[0, :, :, :, -2]) + np.power(10., data_cooling[0, :, :, :, -1]))
    heating_rates = np.log10(np.power(10., data_heating[0, :, :, :, -2]) + np.power(10., data_heating[0, :, :, :, -1]))

    net_rates = np.log10(np.abs(np.power(10., heating_rates) - np.power(10., cooling_rates)))

    axis = get_axis_tables()
    nH_grid = axis[0]
    T_grid = axis[4]
    Z_grid = axis[2]

    f_net_rates = sci.RegularGridInterpolator((T_grid, Z_grid, nH_grid), net_rates, method="linear", bounds_error=False,
                                              fill_value=-30)

    hydrogen_fraction = data.gas.element_mass_fractions.hydrogen
    gas_nH = (data.gas.densities / mh * hydrogen_fraction).to(cm ** -3)
    log_gas_nH = np.log10(gas_nH)
    temperature = data.gas.temperatures
    log_gas_T = np.log10(temperature)
    log_gas_Z = np.log10(data.gas.metal_mass_fractions.value / 0.0133714)

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    net_rates_found = f_net_rates(values_to_int)

    cooling_time = np.log10(3. / 2. * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)

    tff.to("Myr")
    free_fall_time = np.log10(tff.to("Myr"))

    ratio_cooling_time_over_ff_time = cooling_time - free_fall_time

    boxsize = data.metadata.boxsize
    r_gas = ((data.gas.coordinates[:, 0] - boxsize[0] / 2.) ** 2 + (
                data.gas.coordinates[:, 1] - boxsize[1] / 2.) ** 2 + (
                         data.gas.coordinates[:, 2] - boxsize[2] / 2.) ** 2) ** .5
    absolute_velocity = (data.gas.velocities[:, 0] ** 2 + data.gas.velocities[:, 1] ** 2 + data.gas.velocities[:,
                                                                                           2] ** 2) ** .5
    flow_time = r_gas / absolute_velocity
    log_flow_time = np.log10(flow_time.to("Myr"))

    # calculate the enclosed mass from the particles
    radius_used, log_encl_stellar, log_encl_gas = calculate_the_enclosed_masses(data)
    radius_used_log = np.log10(radius_used)
    f_encl_stars = sci.interp1d(radius_used_log, log_encl_stellar)
    f_encl_gas = sci.interp1d(radius_used_log, log_encl_gas)

    encl_mass = Mp_hern(r_gas, 82.3903 * kpc, 1e13 * (1 - 0.01) * Msun) + (
                10 ** f_encl_stars(np.log10(r_gas)) + 10 ** f_encl_gas(np.log10(r_gas))) * Msun

    dynamical_time = np.sqrt(2.) * r_gas / np.sqrt(G * encl_mass / r_gas)
    log_dynamical_time = np.log10(dynamical_time.to("Myr"))

    # prepare things for averaging
    bins = radial_bins.to(r_gas.units)
    masses = data.gas.masses.value
    density = data.gas.densities.value

    mass_avg_c_ff_ratio = normalized_mean(r_gas, ratio_cooling_time_over_ff_time, masses, bins)
    mass_avg_c_flow_ratio = normalized_mean(r_gas, cooling_time - log_flow_time, masses, bins)
    mass_avg_c_dyn_ratio = normalized_mean(r_gas, cooling_time - log_dynamical_time, masses, bins)
    mass_avg_c_time = normalized_mean(r_gas, cooling_time, masses, bins)
    mass_avg_ff_time = normalized_mean(r_gas, free_fall_time, masses, bins)
    mass_avg_dyn_time = normalized_mean(r_gas, log_dynamical_time, masses, bins)
    mass_avg_flow_time = normalized_mean(r_gas, log_flow_time, masses, bins)

    volume_proxy = masses / density
    volume_proxy = volume_proxy

    volume_avg_c_ff_ratio = normalized_mean(r_gas, ratio_cooling_time_over_ff_time, volume_proxy, bins)
    volume_avg_c_flow_ratio = normalized_mean(r_gas, cooling_time - log_flow_time, volume_proxy, bins)
    volume_avg_c_dyn_ratio = normalized_mean(r_gas, cooling_time - log_dynamical_time, volume_proxy, bins)
    volume_avg_c_time = normalized_mean(r_gas, cooling_time, volume_proxy, bins)
    volume_avg_ff_time = normalized_mean(r_gas, free_fall_time, volume_proxy, bins)
    volume_avg_dyn_time = normalized_mean(r_gas, log_dynamical_time, volume_proxy, bins)
    volume_avg_flow_time = normalized_mean(r_gas, log_flow_time, volume_proxy, bins)

    return mass_avg_c_ff_ratio, volume_avg_c_ff_ratio, mass_avg_c_flow_ratio, volume_avg_c_flow_ratio, mass_avg_c_dyn_ratio, volume_avg_c_dyn_ratio, mass_avg_c_time, volume_avg_c_time, mass_avg_ff_time, volume_avg_ff_time, mass_avg_dyn_time, volume_avg_dyn_time, mass_avg_flow_time, volume_avg_flow_time


def calculate_cumulative_cooling_rate(data):
    tff = np.sqrt(3 * np.pi / (32 * G * data.gas.densities))

    data_cooling = get_cooling_rates()
    data_heating = get_heating_rates()

    cooling_rates = np.log10(np.power(10., data_cooling[0, :, :, :, -2]) + np.power(10., data_cooling[0, :, :, :, -1]))
    heating_rates = np.log10(np.power(10., data_heating[0, :, :, :, -2]) + np.power(10., data_heating[0, :, :, :, -1]))

    net_rates = np.log10((np.power(10., cooling_rates) - np.power(10., heating_rates)))
    axis = get_axis_tables()
    nH_grid = axis[0]
    T_grid = axis[4]
    Z_grid = axis[2]

    f_cooling_rates = sci.RegularGridInterpolator((T_grid, Z_grid, nH_grid), net_rates, method="linear",
                                                  bounds_error=False, fill_value=-30)

    hydrogen_fraction = data.gas.element_mass_fractions.hydrogen
    gas_nH = (data.gas.densities / mh * hydrogen_fraction).to(cm ** -3)
    log_gas_nH = np.log10(gas_nH)
    temperature = data.gas.temperatures
    log_gas_T = np.log10(temperature)
    log_gas_Z = np.log10(data.gas.metal_mass_fractions.value / 0.0133714)

    # construct the matrix that we input in the interpolator
    values_to_int = np.zeros((len(log_gas_T), 3))
    values_to_int[:, 0] = log_gas_T
    values_to_int[:, 1] = log_gas_Z
    values_to_int[:, 2] = log_gas_nH

    # get the cooling rates
    cooling_rates_found = f_cooling_rates(values_to_int)

    X = data.gas.element_mass_fractions.hydrogen.value
    # data.gas.masses.convert_to_units("kg")
    gass_mass = data.gas.masses

    print(gass_mass.units / mh.units)
    unit_conversion = gass_mass.units / mh.units
    print(unit_conversion)
    unit_conversion = unit_conversion.as_coeff_unit()[0]
    print(unit_conversion)
    print(type(unit_conversion))
    print(gass_mass)
    print(np.log10(gass_mass))
    print(mh)
    print(mh.value)
    coefficient = np.log10(gass_mass) - np.log10(mh.value) + np.log10(X) + np.log10(unit_conversion)
    print("Extra output!!!!")
    print(coefficient)

    # calculate the cooling rate in erg/s
    log_cooling_rates_erg_per_s = cooling_rates_found + log_gas_nH + coefficient
    print(cooling_rates_found)
    print(np.max(cooling_rates_found), np.min(cooling_rates_found), np.average(cooling_rates_found))
    print(log_cooling_rates_erg_per_s)
    print(np.max(log_cooling_rates_erg_per_s), np.min(log_cooling_rates_erg_per_s),
          np.average(log_cooling_rates_erg_per_s))
    cooling_rates_erg_per_s = np.power(10., log_cooling_rates_erg_per_s)
    print(cooling_rates_erg_per_s)
    print(np.max(cooling_rates_erg_per_s), np.min(cooling_rates_erg_per_s), np.average(cooling_rates_erg_per_s))

    boxsize = data.metadata.boxsize
    r_gas = ((data.gas.coordinates[:, 0] - boxsize[0] / 2.) ** 2 + (
                data.gas.coordinates[:, 1] - boxsize[1] / 2.) ** 2 + (
                         data.gas.coordinates[:, 2] - boxsize[2] / 2.) ** 2) ** .5
    r_gas.to("Mpc")
    log_r_gas = np.log10(r_gas.value) - 3.

    hist_result, bins = np.histogram(log_r_gas, weights=cooling_rates_erg_per_s, range=[-4, log_r_max], bins=200)
    hist_result_cum = np.cumsum(hist_result)

    mask = (data.gas.temperatures > 1e6 * unyt.K)
    hist_result2, bins = np.histogram(log_r_gas[mask], weights=cooling_rates_erg_per_s[mask], range=[-4, log_r_max],
                                      bins=200)
    count, bins = np.histogram(log_r_gas, range=[-4, log_r_max], bins=200)

    mask2 = (data.gas.subgrid_temperatures > 1e4 * unyt.K)
    hist_result3, bins = np.histogram(log_r_gas[mask2], weights=cooling_rates_erg_per_s[mask2], range=[-4, log_r_max],
                                      bins=200)

    hist_result2_cum = np.cumsum(hist_result2)
    hist_result3_cum = np.cumsum(hist_result3)
    hist_result2_cum_inversed = np.flip(np.cumsum(np.flip(hist_result2)))
    return (bins[:-1] + bins[
                        1:]) / 2., hist_result2_cum, hist_result_cum, hist_result3_cum, count, hist_result2_cum_inversed


def calculate_cumulative_x_ray_flux(data):
    hydrogen_fraction = data.gas.element_mass_fractions.hydrogen
    # Compute number density
    data_n = np.log10(hydrogen_fraction * data.gas.densities.to(unyt.g * cm ** -3) / mh)

    # get temperature
    data_T = np.log10(data.gas.temperatures.value)

    # interpolate
    data.gas.log_X_ray_emissivity = interpolate_X_Ray.interpolate_X_Ray(data_n, data_T, data.gas.element_mass_fractions,
                                                                        fill_value=-50.)

    gas_nH = (data.gas.densities / mh * hydrogen_fraction).to(cm ** -3)
    log_gas_nH = np.log10(gas_nH)
    log_gas_Z = np.log10(data.gas.metal_mass_fractions.value / 0.0133714)

    # get individual elements
    X = data.gas.element_mass_fractions.hydrogen.value
    gass_mass = data.gas.masses
    coefficient = np.log10(gass_mass / mh) + np.log10(X)

    data.gas.X_ray_emissivity = np.power(10, data.gas.log_X_ray_emissivity) * unyt.erg / (unyt.s * unyt.cm ** 3)

    data.gas.Xray_erg_per_s = (data.gas.X_ray_emissivity * (data.gas.masses / data.gas.densities)).to(unyt.erg / unyt.s)

    boxsize = data.metadata.boxsize
    r_gas = ((data.gas.coordinates[:, 0] - boxsize[0] / 2.) ** 2 + (
                data.gas.coordinates[:, 1] - boxsize[1] / 2.) ** 2 + (
                         data.gas.coordinates[:, 2] - boxsize[2] / 2.) ** 2) ** .5
    r_gas.to("Mpc")
    log_r_gas = np.log10(r_gas.value) - 3.

    hist_result, bins = np.histogram(log_r_gas, weights=data.gas.Xray_erg_per_s, range=[-4, log_r_max], bins=200)
    hist_result_cum = np.cumsum(hist_result)

    mask = (data.gas.temperatures > (10 ** 6.5) * unyt.K)
    hist_result2, bins = np.histogram(log_r_gas[mask], weights=data.gas.Xray_erg_per_s[mask], range=[-4, log_r_max],
                                      bins=200)
    count, bins = np.histogram(log_r_gas, range=[-4, log_r_max], bins=200)

    hist_result2_cum = np.cumsum(hist_result2)
    hist_result2_cum_inversed = np.flip(np.cumsum(np.flip(hist_result2)))
    return (bins[:-1] + bins[1:]) / 2., hist_result2_cum, hist_result_cum, count, hist_result2_cum_inversed
