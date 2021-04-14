from register import (
    calibration_zooms,
    completed_runs,
    zooms_register,
)


def query_exlzooms() -> None:
    incomplete_runs = calibration_zooms.get_incomplete_run_directories()
    print((
        "\n"
        "The following simulations were found with directory set-up, "
        "but missing snapshots or stf sub-directories. They are "
        "likely not yet launched or incomplete and were not appended "
        "to the master register."
    ))
    for i in incomplete_runs:
        print(f"[!] -> {i:s}")

    print(f"\n{' Zoom register ':-^40s}")
    for i in completed_runs:
        print(i)

    print(f"\n{' Test: redshift data (z = 0) ':-^40s}")
    print(zooms_register[0].get_redshift())

    print(f"\n{' Test: redshift data (z = 0.1) ':-^40s}")
    print(zooms_register[0].get_redshift(-3))

    advanced_search = input((
        "Press `y` to initialise the advanced search on incomplete runs. "
        "Press any other key to quit.\t--> "
    ))
    if advanced_search == 'y':
        calibration_zooms.analyse_incomplete_runs()


if __name__ == '__main__':
    query_exlzooms()