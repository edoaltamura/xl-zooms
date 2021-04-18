import unittest
import contextlib
from tqdm import tqdm

from register import (
    calibration_zooms,
    completed_runs,
    zooms_register,
)


class TestRegister(unittest.TestCase):
    def run_test(self):
        incomplete_runs = calibration_zooms.get_incomplete_run_directories()
        for _ in tqdm(incomplete_runs, desc='[Incomplete runs]'):
            pass
        print('Done.')

        print(f"\n{' Zoom register ':-^40s}")
        for _ in tqdm(completed_runs, desc='[Complete runs]'):
            pass
        print('Done.')

        print(f"\n{' Test: redshift data (z = 0) ':-^40s}")
        print(zooms_register[0].get_redshift())

        print(f"\n{' Test: redshift data (z = 0.1) ':-^40s}")
        print(zooms_register[0].get_redshift(-3))

        print("Advanced search (incl SLURM) testing byt not printing...")
        with contextlib.redirect_stdout(None):
            calibration_zooms.analyse_incomplete_runs()
        print('Done.')
        self.assertEqual(0, 0)


if __name__ == '__main__':
    unittest.main()
