import sys
import os
import subprocess
import re
import h5py
import yaml
import numpy as np
from scipy.spatial import distance
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from MakeGrid import *
from ParallelFunctions import repartition
from MakeParamFile import (
    make_submit_file_ics,
    make_submit_file_gadget,
    make_submit_file_swift,
    make_param_file_ics,
    make_param_file_swift
)

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class ParticleLoad:

    def __init__(self, param_file, randomize=False, only_calc_ntot=False,
                 check_gaps=False, save_hdf5=False, do_gadget=False, do_swift=True,
                 do_gadget4=False, save_data=True):

        self.randomize = randomize
        self.only_calc_ntot = only_calc_ntot
        self.check_gaps = check_gaps
        self.save_hdf5 = save_hdf5
        self.do_gadget = do_gadget
        self.do_swift = do_swift
        self.do_gadget4 = do_gadget4
        self.save_data = save_data

        # Save directories.
        self.pl_dir = '/cosma7/data/dp004/dc-alta2/xl-zooms/ics/particle_load/'
        self.gadget_dir = '/cosma7/data/dp004/dc-alta2/xl-zooms/ics/'
        self.swift_dir = '/cosma7/data/dp004/dc-alta2/xl-zooms/ics/swift_runs/'
        self.ic_dir = '/cosma7/data/dp004/dc-alta2/xl-zooms/ics/ic_gen/run/'
        self.gadget4_dir = '/cosma7/data/dp004/dc-alta2/xl-zooms/ics/'

        # Read the parameter file.
        self.read_param_file(param_file)

        # Generate particle load.
        self.make_particle_load()

    def read_param_file(self, param_file):
        """ Read in parameters for run. """
        if comm_rank == 0:
            params = yaml.load(open(param_file))
        else:
            params = None
        params = comm.bcast(params, root=0)

        # Params that are required.
        required_params = ['box_size', 'n_particles', 'glass_num', 'f_name',
                           'panphasian_descriptor', 'ndim_fft_start', 'is_zoom', 'which_cosmology']
        for att in required_params:
            assert att in params.keys(), 'Need to have %s as a param' % att

        # Some dummy placers.
        self.coords = np.array([0., 0., 0.])
        self.radius = 0.
        self.mask_file = None  # Use a precomputed mask for glass particles.
        self.all_grid = False  # Only make from grids (no glass particles).
        self.n_species = 1  # Number of DM species.

        self.constraint_phase_descriptor = '%dummy'
        self.constraint_phase_descriptor_path = '%dummy'
        self.constraint_phase_descriptor_levels = '%dummy'

        self.constraint_phase_descriptor2 = '%dummy'
        self.constraint_phase_descriptor_path2 = '%dummy'
        self.constraint_phase_descriptor_levels2 = '%dummy'

        # Params for hi res grid.
        self.skin_reduce_factor = 1 / 8.  # What factor do high res skins reduce by.
        self.min_num_per_cell = 1  # Min number of particles in high res cell (must be cube).
        self.radius_factor = 1.
        self.glass_buffer_cells = 0  # Number of buffer cells on each side (must be even, eg. 2 = 1 on each side)
        self.ic_region_buffer_frac = 1.25  # 25% (buffer for FFT grid during ICs).

        # Default starting and finishing redshifts.
        self.starting_z = 127.0
        self.finishing_z = 0.0

        # Is DM only? (Only important for softenings).
        self.dm_only = False

        # What type of IDs to use.
        self.use_ph_ids = True
        self.nbit = 21  # 14 for EAGLE

        # If a zoom, use multigrid IC grid?
        self.multigrid_ics = True

        # Info for submit files.
        self.n_cores_ic_gen = 28
        self.n_cores_gadget = 32
        self.n_nodes_swift = 1
        self.num_hours_ic_gen = 24
        self.num_hours_swift = 72

        # Params for outer low res particles
        self.min_nq = 30
        self._max_nq = 100

        # For special set of params.
        self.template_set = 'default'

        # Is this a slab simulation?
        self.is_slab = False

        # Use glass files to surround the high res region rather than grids?
        self.grid_also_glass = True
        self.glass_files_dir = '../glass_files/'

        # Softening for zooms.
        self.softening_ratio_background = 0.02  # 1/50 M-P-S.

        # Default GADGET executable.
        self.gadget_exec = 'P-Gadget3-DMO-NoSF'

        # Relace with param file values.
        for att in params.keys():
            setattr(self, att, params[att])

        # Assign cosmology.
        if self.which_cosmology == 'Planck2013':
            self.Omega0 = 0.307
            self.OmegaLambda = 0.693
            self.OmegaBaryon = 0.04825
            self.HubbleParam = 0.6777
            self.Sigma8 = 0.8288
            self.linear_ps = 'extended_planck_linear_powspec'

        elif self.which_cosmology == 'Planck2018':
            self.Omega0 = 0.3111
            self.OmegaLambda = 0.6889
            self.OmegaBaryon = 0.04897
            self.HubbleParam = 0.6766
            self.Sigma8 = 0.8102
            self.linear_ps = 'EAGLE_XL_powspec_18-07-2019.txt'
        else:
            raise ValueError("Invalid cosmology")

        # Make sure coords is numpy array.
        self.coords = np.array(self.coords)

        # Sanity check.
        assert np.true_divide(self.n_particles, self.glass_num) % 1 < 1e-6, \
            'Number of particles must divide into glass_num'

    def compute_masses(self):
        """ For the given cosmology, compute the total DM mass for the given volume. """
        if comm_rank == 0:
            cosmo = FlatLambdaCDM(H0=self.HubbleParam * 100., Om0=self.Omega0, Ob0=self.OmegaBaryon)
            rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc ** 3)

            M_tot_dm_dmo = self.Omega0 * rho_crit.value * (self.box_size / self.HubbleParam) ** 3
            M_tot_dm = (self.Omega0 - self.OmegaBaryon) * rho_crit.value * (self.box_size / self.HubbleParam) ** 3
            M_tot_gas = self.OmegaBaryon * rho_crit.value * (self.box_size / self.HubbleParam) ** 3

            dm_mass = M_tot_dm / self.n_particles
            dm_mass_dmo = M_tot_dm_dmo / self.n_particles
            gas_mass = M_tot_gas / self.n_particles

            print('Dark matter particle mass (if DMO): %.3g Msol (%.3g 1e10 Msol/h)' \
                  % (dm_mass_dmo, dm_mass_dmo * self.HubbleParam / 1.e10))
            print('Dark matter particle mass: %.3g Msol (%.3g 1e10 Msol/h)' \
                  % (dm_mass, dm_mass * self.HubbleParam / 1.e10))
            print('Gas particle mass: %.3g Msol (%.3g 1e10 Msol/h)' % \
                  (gas_mass, gas_mass * self.HubbleParam / 1.e10))
        else:
            M_tot_dm_dmo = None

        M_tot_dm_dmo = comm.bcast(M_tot_dm_dmo)
        self.total_box_mass = M_tot_dm_dmo

    def init_high_res(self, offsets, cell_nos, L):
        """ Initialize the high resolution grid with the primary high resolution particles. """
        cell_types = np.ones(len(offsets), dtype='i4') * -1  # Should all get overwritten.

        # For zoom simulations.
        if self.is_zoom:
            # Using a mask file.
            if self.mask_file is not None:
                if comm_rank == 0: print('Loading mask file %s' % self.mask_file)
                # Rescale mask coords into grid coords.
                self.mask_coords *= (self.radius_in_cells / self.radius)

                if comm_rank == 0:
                    print('Mask coords x>=%.5f <=%.5f' % (self.mask_coords[:, 0].min(),
                                                          self.mask_coords[:, 0].max()))
                    print('Mask coords y>=%.5f <=%.5f' % (self.mask_coords[:, 1].min(),
                                                          self.mask_coords[:, 1].max()))
                    print('Mask coords z>=%.5f <=%.5f' % (self.mask_coords[:, 2].min(),
                                                          self.mask_coords[:, 2].max()))

                # Find out which cells in our grid will be glass.
                get_assign_mask_cells(cell_types, self.mask_coords, offsets,
                                      self.mask_grid_cell_width * (self.radius_in_cells / self.radius),
                                      cell_nos, L)

                # Fill the rest with degrading resolution grid cells.
                self.add_high_res_skins(cell_nos, cell_types, L)

            # Using a lagrangian sphere around passed coordinates.
            else:
                raise Exception("Does this work without a mask?")
                # if self.is_slab:
                #    tmp_mask = np.where(np.abs(centers[:,2]) <= (self.slab_width_cells/2.)+1e-4)
                #    cell_types[tmp_mask] = 0
                # else:
                #    if comm_rank == 0: print('Computing distances to each grid cell...')
                #    dists = np.linalg.norm(centers, axis=1)
                #    if comm_rank == 0:
                #        print('Sphere rad %.2f Mpc/h center %s Mpc/h (%.2f Grid cells)'\
                #            %(self.radius, self.coords, self.radius_in_cells))
                #    mask = np.where(dists <= self.radius_factor*self.radius_in_cells)
                #    cell_types[mask] = 0
                #        
                #    # Add skins around the glass cells.
                #    self.add_high_res_skins(cell_nos, cell_types, L)    
        else:
            # All cells are glass.
            cell_types = np.ones(len(offsets), dtype='i4') * 0

        return cell_types

    def add_high_res_skins(self, cell_nos, cell_types, L):
        """ Add degrading skins around glass cells in high res grid. """

        if comm_rank == 0:
            print('Adding skins around glass cells in high res region...')
        mask = np.where(cell_types == -1)
        num_to_go = comm.allreduce(len(mask[0]))
        this_type = 0
        count = 0

        # Loop until we have no more cells to fill.
        while num_to_go > 0:
            skin_cells = np.unique(get_find_skin_cells(cell_types,
                                                       cell_nos, L, this_type))
            skin_cells_counts = comm.allgather(len(skin_cells))
            skin_cells_disps = np.cumsum(skin_cells_counts) - skin_cells_counts
            if comm_rank == 0:
                skin_cells_all = np.empty(np.sum(skin_cells_counts), dtype='i4')
            else:
                skin_cells_all = None
            comm.Gatherv(skin_cells, [skin_cells_all, skin_cells_counts, skin_cells_disps,
                                      MPI.INT], root=0)
            if comm_rank == 0:
                skin_cells = np.unique(skin_cells_all)
            else:
                skin_cells = None
            skin_cells = comm.bcast(skin_cells)

            idx = np.where(np.in1d(cell_nos, skin_cells))[0]
            idx2 = np.where(cell_types[idx] == -1)
            idx3 = idx[idx2]
            cell_types[idx3] = this_type + 1
            mask = np.where(cell_types == -1)
            num_to_go = comm.allreduce(len(mask[0]))
            this_type += 1
            count += 1

        if comm_rank == 0:
            print('Added %i skins.' % count)

    def rescale(self, x, x_min_old, x_max_old, x_min_new, x_max_new):
        """ Rescale an array of numbers to a new min max. """
        return ((x_max_new - x_min_new) / (x_max_old - x_min_old)) * (x - x_max_old) + x_max_new

    def find_nearest_cube(self, num):
        """ Find the nearest number that has a cube root. """
        return int(np.ceil(num ** (1 / 3.)) ** 3.)

    def find_nq_slab(self, suggested_nq, slab_width, eps=1.e-4):

        # What is half a slab width? (our starting point)/
        half_slab = slab_width / 2.

        # Dict to save info.
        self.nq_info = {'diff': 1.e20, 'n_tot_lo': 0}

        # What are we starting from?
        if comm_rank == 0:
            print(f'Computing slab nq: half_slab={half_slab:.2f} suggested_nq={suggested_nq:d}')

        # You start with a given suggested nq, then you remove nq_reduce from it at each level.
        # This tries multiple nq_reduce values.
        for nq_reduce in np.arange(1, 10):

            # Loop over an arbitary number of starting nqs to test them (shouldn't ever need).
            for i in range(200):

                # If this takes me >50% away from the suggested nq, don't bother.
                if np.true_divide(suggested_nq - i, suggested_nq) < 0.5:
                    break

                # Reset all values.
                offset = half_slab  # Starting from the edge of the slab.
                this_nq = suggested_nq - i  # Trying this starting nq.
                starting_nq = suggested_nq - i
                nlev_slab = 0  # Counting the number of levels.
                n_tot_lo = 0  # Counting the total number of low res particles.

                # Iterate levels for this starting nq until you reach the edge of the box.
                while True:
                    # Cell length at this level.
                    m_int_sep = self.box_size / float(this_nq)

                    # Add this level.
                    offset += m_int_sep
                    nlev_slab += 1

                    # How close are we to the edge.
                    diff = (offset - self.box_size / 2.) / self.box_size

                    # If adding the level has gone outside the box, then stop.
                    # Or if we are really close to the edge.
                    if (offset > self.box_size / 2.) or np.abs(diff) <= 1.e-3:

                        # Try to make it a bit better.

                        # First go back to the previous level.
                        offset -= m_int_sep

                        # Loop over a range of new nq options for the last level/
                        for extra in np.arange(-nq_reduce, nq_reduce, 1):
                            # Try a new nq for the last level.
                            this_nq += extra
                            m_int_sep = self.box_size / float(this_nq)
                            diff = (offset + m_int_sep - self.box_size / 2.) / self.box_size

                            # Have we found a new best level?
                            if np.abs(diff) < np.abs(self.nq_info['diff']):
                                self.nq_info['diff'] = diff
                                self.nq_info['starting_nq'] = starting_nq
                                self.nq_info['finishing_nq'] = this_nq - extra
                                self.nq_info['nq_reduce'] = nq_reduce
                                self.nq_info['extra'] = extra
                                self.nq_info['nlev_slab'] = nlev_slab
                                if (nlev_slab - 1) % comm_size == comm_rank:
                                    n_tot_lo += 2 * this_nq ** 2
                                self.nq_info['n_tot_lo'] = n_tot_lo
                                self.nq_info['dv_slab'] = -1.0 * (
                                            (offset + m_int_sep - self.box_size / 2.) * self.box_size ** 2.) / \
                                                          self.nq_info['finishing_nq'] ** 2.

                                # Reset for next try.
                                if (nlev_slab - 1) % comm_size == comm_rank: n_tot_lo -= 2 * this_nq ** 2

                            # Reset for next try.
                            this_nq -= extra
                        break
                    else:
                        if (nlev_slab - 1) % comm_size == comm_rank:
                            n_tot_lo += 2 * this_nq ** 2

                    # Compute nq for next level.
                    this_nq -= nq_reduce

                    # We've gotten too small. 
                    if this_nq < 10:
                        break

        if comm_rank == 0:
            print(
                '[Rank %i] Best nq: starting_nq=%i, finishing_nq=%i, extra=%i, diff=%.10f, nlev_slab=%i, nq_reduce=%i, dv_slab=%.10f n_tot_lo=%i' % \
                (comm_rank, self.nq_info['starting_nq'], self.nq_info['finishing_nq'], self.nq_info['extra'],
                 self.nq_info['diff'], self.nq_info['nlev_slab'], self.nq_info['nq_reduce'], self.nq_info['dv_slab'],
                 self.nq_info['n_tot_lo']))
        assert np.abs(self.nq_info['diff']) <= eps, 'Could not find good nq'

        return self.nq_info['n_tot_lo']

    def find_nq(self, side, suggested_nq, eps=0.01):
        """ Estimate what the best value of nq should be. """

        self.nq_info = {'diff': 1.e20}
        lbox = 1. / side
        if comm_rank == 0:
            print('Looking for best nq, suggested_nq=%i' % (suggested_nq))

        # Loop over a range of potential nq's.
        for nq in np.arange(suggested_nq - 5, suggested_nq + 5, 1):
            if nq < 10: continue
            found_good = 0

            # Loop over a range of extras.
            for extra in range(-10, 10, 1):
                if nq + extra < 10:
                    continue

                # For this nq and extra, what volume would the particles fill.
                total_volume, nlev = get_guess_nq(lbox, nq, extra, comm_rank, comm_size)
                total_volume = comm.allreduce(total_volume)

                # How does this volume compare to the volume we need to fill?
                diff = np.abs(1 - (total_volume / (lbox ** 3. - 1. ** 3)))

                if diff < self.nq_info['diff']:
                    self.nq_info['diff'] = diff
                    self.nq_info['nq'] = nq
                    self.nq_info['extra'] = extra
                    self.nq_info['nlev'] = nlev
                    self.nq_info['total_volume'] = total_volume

        assert self.nq_info['diff'] <= eps, 'Did not find good nq. (diff = %.6f)' % (self.nq_info['diff'])

        # Compute low res particle number for this core.
        n_tot_lo = 0
        for l in range(self.nq_info['nlev']):
            if l % comm_size != comm_rank:
                continue
            if l == self.nq_info['nlev'] - 1:
                n_tot_lo += (self.nq_info['nq'] - 1 + self.nq_info['extra']) ** 2 * 6 + 2
            else:
                n_tot_lo += (self.nq_info['nq'] - 1) ** 2 * 6 + 2
        self.nq_info['n_tot_lo'] = n_tot_lo

        # Report what we found.
        if comm_rank == 0:
            print('[Rank %i] Best nq: suggested_nq=%i, nq=%i, extra=%i, diff=%.10f, nlev=%i, n_tot_lo=%i' % \
                  (comm_rank, suggested_nq, self.nq_info['nq'], self.nq_info['extra'],
                   self.nq_info['diff'], self.nq_info['nlev'], self.nq_info['n_tot_lo']))

        return n_tot_lo

    def find_ntot(self, num_in_glass_cell, cell_types, side, max_cells,
                  slab_width, high_res_region_max):
        """ For this core, work out how many particles it will work with. """

        # Number of high resolution particles this core will have.
        self.cell_info = {'type': [], 'num_particles_per_cell': [],
                          'num_cells': [], 'particle_mass': []}
        n_tot_hi = 0
        tot_num_glass_particles = 0
        tot_num_grid_particles = 0

        # Loop over each cell type/particle mass.
        for i in np.unique(cell_types):
            num_cells = len(np.where(cell_types == i)[0])

            self.cell_info['type'].append(i)
            this_num_cells = len(np.where(cell_types == i)[0])
            self.cell_info['num_cells'].append(this_num_cells)

            # Glass particles (type 0).
            if i == 0:
                self.cell_info['num_particles_per_cell'].append(num_in_glass_cell)
                n_tot_hi += this_num_cells * num_in_glass_cell
                tot_num_glass_particles += this_num_cells * num_in_glass_cell
            # Grid particles (type > 0).
            else:
                # Desired number of particles in this level of grid.
                desired_no = np.maximum(self.min_num_per_cell,
                                        int(np.ceil(num_in_glass_cell * \
                                                    np.true_divide(self.skin_reduce_factor, i))))

                if self.grid_also_glass:
                    # Find glass file with the closest number of particles.
                    num_in_grid_cell = self.find_nearest_glass_file(desired_no)
                else:
                    # Find nearest cube to this number, for grid.
                    num_in_grid_cell = self.find_nearest_cube(desired_no)

                self.cell_info['num_particles_per_cell'].append(num_in_grid_cell)
                n_tot_hi += this_num_cells * num_in_grid_cell
                tot_num_grid_particles += this_num_cells * num_in_grid_cell

        # Total number of glass particles.
        self.n_tot_glass_part = comm.allreduce(tot_num_glass_particles)
        # Total number of grid/lower mass glass particles.
        self.n_tot_grid_part = comm.allreduce(tot_num_grid_particles)
        # The number of glass particles if they filled the high res grid.
        self.n_tot_glass_part_equiv = num_in_glass_cell * comm.allreduce(len(cell_types))
        # How many particles in the lowest mass resolution cells in the high res grid.
        num_lowest_res = comm.allreduce(np.min(self.cell_info['num_particles_per_cell']), op=MPI.MIN)
        # The number of the most massive grid/lower mass glass particles if they filled the high res grid.
        self.n_tot_grid_part_equiv = num_lowest_res * comm.allreduce(len(cell_types))
        # How many levels of resolution are in the high res grid for this rank.
        self.my_cell_types = np.unique(cell_types)

        # Number of low resolution particles this core will have.
        if self.is_zoom:
            if self.is_slab:
                # Starting nq is equiv of double the mass of the most massive grid particles.
                suggested_nq = int(num_lowest_res ** (1 / 3.) * max_cells * 0.5)
                n_tot_lo = self.find_nq_slab(suggested_nq, slab_width)
            else:
                # Dont want boundary particles more massive than high-res region particles.
                self.max_nq = np.minimum(int(np.floor(self.n_tot_grid_part_equiv ** (1 / 3.))),
                                         self._max_nq)

                # Starting nq is equiv of double the mass of the most massive grid particles.
                suggested_nq = int(self.n_tot_grid_part_equiv ** (1 / 3.) * 0.5)
                if comm_rank == 0:
                    print("num_lowest_res=%i (%.2f cubed)" % (num_lowest_res, num_lowest_res ** (1 / 3.)))
                    print("n_tot_grid_part_equiv=%i (%.2f cubed) (%.2f particles per [cMpc/h]**3)" \
                          % (self.n_tot_grid_part_equiv, self.n_tot_grid_part_equiv ** (1 / 3.),
                             np.true_divide(self.n_tot_grid_part_equiv, high_res_region_max ** 3.)))
                    print("The nq I wanted was %i" % suggested_nq)
                if suggested_nq < 30: suggested_nq = 30
                if suggested_nq > 75: suggested_nq = 75

                # Compute nq
                n_tot_lo = self.find_nq(side, suggested_nq)
        else:
            n_tot_lo = 0

        return n_tot_hi, n_tot_lo

    def find_nearest_glass_file(self, num):
        """ In the glass files folder, which one has the closest number of particles. """
        files = os.listdir(self.glass_files_dir)
        files = np.array([int(x.split('_')[2]) for x in files if 'ascii' in x], dtype='i8')
        idx = np.abs(files - num).argmin()
        return files[idx]

    def populate_levels(self, cell_length, cell_types, offsets, glass,
                        ntot, coords_x, coords_y, coords_z, masses):
        """ This populates the type 0 (glass) and type >0 (grid) cells of the high res grid. """

        # Compute the masses of each particle type in the high res-grid.
        for num_per_cell in self.cell_info['num_particles_per_cell']:
            self.cell_info['particle_mass'].append(((cell_length / num_per_cell ** (1 / 3.)) / self.box_size) ** 3.)
        for att in self.cell_info.keys():
            self.cell_info[att] = np.array(self.cell_info[att])
        all_particle_masses = np.unique(np.concatenate(comm.allgather(self.cell_info['particle_mass'])))
        mask = all_particle_masses != np.min(all_particle_masses)
        if self.is_zoom:
            self.min_grid_mass = np.min(all_particle_masses[mask])
            self.max_grid_mass = np.max(all_particle_masses[mask])
        self.glass_particle_mass = np.min(all_particle_masses)
        if comm_rank == 0:
            if self.is_zoom:
                print('Glass mass %.8g, min grid mass %.8g, max grid mass %.8g' \
                      % (self.glass_particle_mass, self.min_grid_mass, self.max_grid_mass))
            else:
                print('Glass mass %.8g' % (self.glass_particle_mass))

        # Loop over each cell type and fill up the grid.
        cell_offset = 0
        for i in range(len(self.cell_info['type'])):
            mask = np.where(cell_types == self.cell_info['type'][i])
            assert len(mask[0]) > 0, 'Dont have types that I should.'

            # Glass particles.
            if self.cell_info['type'][i] == 0:
                if self.all_grid:
                    get_populated_grid(offsets[mask],
                                       self.generate_uniform_grid(self.cell_info['num_particles_per_cell'][i]),
                                       coords_x, coords_y, coords_z, cell_offset)
                else:
                    get_populated_grid(offsets[mask], np.c_[glass[self.glass_num]['x'],
                                                            glass[self.glass_num]['y'], glass[self.glass_num]['z']],
                                       coords_x, coords_y, coords_z, cell_offset)
            # Grid particles:
            else:
                if self.grid_also_glass:
                    get_populated_grid(offsets[mask],
                                       np.c_[glass[self.cell_info['num_particles_per_cell'][i]]['x'],
                                             glass[self.cell_info['num_particles_per_cell'][i]]['y'],
                                             glass[self.cell_info['num_particles_per_cell'][i]]['z']],
                                       coords_x, coords_y, coords_z, cell_offset)
                else:
                    get_populated_grid(offsets[mask],
                                       self.generate_uniform_grid(self.cell_info['num_particles_per_cell'][i]),
                                       coords_x, coords_y, coords_z, cell_offset)

            masses[cell_offset:cell_offset + len(mask[0]) * self.cell_info['num_particles_per_cell'][i]] \
                = self.cell_info['particle_mass'][i]

            # Print info for this cell type.
            print('[%i: Type %i] n=%i (%.2f^3) in %i cells (%i/cell) m=%.4f (%.2g Msol/h)' % \
                  (comm_rank, self.cell_info['type'][i],
                   len(mask[0]) * self.cell_info['num_particles_per_cell'][i],
                   (len(mask[0]) * self.cell_info['num_particles_per_cell'][i]) ** (1 / 3.), len(mask[0]),
                   self.cell_info['num_particles_per_cell'][i],
                   np.log10(self.cell_info['particle_mass'][i]),
                   self.cell_info['particle_mass'][i] * self.total_box_mass))

            cell_offset += len(mask[0]) * self.cell_info['num_particles_per_cell'][i]

        # Print total number info.
        reduced_n_tot = comm.allreduce(ntot)
        reduced_n_tot_cells = comm.allreduce(np.sum(self.cell_info['num_cells']))
        comm.barrier()
        if comm_rank == 0:
            print('Total %i particles in high res region (%.2f cubed) in %i cells (%.2f cubed)' % \
                  (reduced_n_tot, (reduced_n_tot) ** (1 / 3.),
                   reduced_n_tot_cells, (reduced_n_tot_cells) ** (1 / 3.)))
            print('Equiv num of glass particles if they filled the grid %i (%.2f, eqv nq)' % \
                  (self.n_tot_glass_part_equiv, (self.n_tot_glass_part_equiv) ** (1 / 3.)))
            print('Equiv num of grid particles if they filled the grid %i (%.2f, eqv nq)' % \
                  (self.n_tot_grid_part_equiv, (self.n_tot_grid_part_equiv) ** (1 / 3.)))
            print('Number of glass particles if they filled the box %i (%.2f cubed)' % \
                  (self.n_particles, (self.n_particles) ** (1 / 3.)))

    def load_glass_file(self, num):
        """ Load the glass file for high resolution particles. """
        if not self.all_grid:
            glass = np.loadtxt(self.glass_files_dir + 'ascii_glass_%i' % num,
                               dtype={'names': ['x', 'y', 'z'], 'formats': ['f8', 'f8', 'f8']},
                               skiprows=1)
            if comm_rank == 0:
                print('Loaded glass file, %i particles in file.' % num)
        else:
            glass = None

        return glass

    def load_mask_file(self):
        """ Load mask file that outlines region. """
        self.mask_grid_cell_width = None
        if self.mask_file is not None:
            if comm_rank == 0:
                print('\n------ Loading mask file ------')
                f = h5py.File(self.mask_file, 'r')
                self.mask_coords = np.array(f['Coordinates'][...], dtype='f8')
                self.coords = f['Coordinates'].attrs.get("com_coords")  # Center of high res.
                bounding_box = [2. * np.maximum(f['Coordinates'].attrs.get("xlen_lo"),
                                                f['Coordinates'].attrs.get("xlen_hi")),
                                2. * np.maximum(f['Coordinates'].attrs.get("ylen_lo"),
                                                f['Coordinates'].attrs.get("ylen_hi")),
                                2. * np.maximum(f['Coordinates'].attrs.get("zlen_lo"),
                                                f['Coordinates'].attrs.get("zlen_hi"))]
                self.radius = np.max(bounding_box) / 2.
                self.mask_high_res_volume = f['Coordinates'].attrs.get("high_res_volume")
                self.mask_grid_cell_width = f['Coordinates'].attrs.get("grid_cell_width")
                f.close()
                print('Loaded: %s' % self.mask_file)
                print('Mask bounding box = %s' % bounding_box)
            else:
                for att in ['mask_coords', 'coords', 'radius', 'mask_high_res_volume',
                            'mask_grid_cell_width']:
                    setattr(self, att, None)
                bounding_box = None
            for att in ['mask_coords', 'coords', 'radius', 'mask_high_res_volume',
                        'mask_grid_cell_width']:
                setattr(self, att, comm.bcast(getattr(self, att)))
            bounding_box = comm.bcast(bounding_box)
        else:
            bounding_box = [2. * self.radius, 2. * self.radius, 2. * self.radius]
        return bounding_box

    def make_particle_load(self):

        if comm_rank == 0:
            print('\n-------------------------')
            print('Generating particle load.')
            print('-------------------------\n')

        # Get particle masses of high resolution particles.
        self.compute_masses()
        eps_dm, eps_baryon, eps_dm_physical, eps_baryon_physical = \
            self.compute_softning(verbose=True)

        # Load mask file.
        bounding_box = self.load_mask_file()

        # |----------------------------------------|
        # | First compute the high resolution grid |
        # |----------------------------------------|

        # Number of glass cells that fill a box (on a side).
        n_cells = int(np.rint((self.n_particles / self.glass_num) ** (1 / 3.)))
        assert n_cells ** 3 * self.glass_num == self.n_particles, 'Error creating high res cell sizes'
        cell_length = np.true_divide(self.box_size, n_cells)  # Mpc/h
        if comm_rank == 0:
            print('\n------ High res grid ------')
            print('A glass cell length is %.4f Mpc/h' % (cell_length))
        if self.is_zoom:
            # X,Y,Z num of cells in high res grid.

            # Number of buffer cells around glass particles (one on each side).
            buf = self.glass_buffer_cells

            # Dimensions of high resolution grid in glass cells.
            n_cells_high = np.array([
                np.minimum(buf + int(np.ceil(bounding_box[0] / cell_length)), n_cells),
                np.minimum(buf + int(np.ceil(bounding_box[1] / cell_length)), n_cells),
                np.minimum(buf + int(np.ceil(bounding_box[2] / cell_length)), n_cells)],
                dtype='i4')
        else:
            # Glass cells makes up whole box.
            n_cells_high = np.array([n_cells, n_cells, n_cells])

        if self.is_zoom:
            self.radius_in_cells = np.true_divide(self.radius, cell_length)

            # Make sure grid can accomodate the radius factor.
            # if self.is_slab == False:
            #    sys.exit('radius factor not tested')
            #    n_cells_high += \
            #        2*int(np.ceil(self.radius_factor*self.radius_in_cells-self.radius_in_cells))

            # What's the width of the slab in cells.
            if self.is_slab:
                self.slab_width_cells = \
                    np.minimum(int(np.ceil(bounding_box[2] / cell_length)), n_cells)
                if comm_rank == 0:
                    print('Number of cells for the width of slab = %i' % self.slab_width_cells)

        # Compute grid dimensions.
        tmp_mask = np.where(n_cells_high > n_cells)
        n_cells_high[tmp_mask] = n_cells
        max_cells = np.max(n_cells_high)
        min_cells = np.min(n_cells_high)

        if self.is_slab == False:
            # Default of high resolution grid is a cube.
            n_cells_high = [max_cells, max_cells, max_cells]
        else:
            # For a slab keep the asymmetry.
            tmp_mask = np.where(n_cells_high == n_cells)[0]
            assert len(tmp_mask) == 2, 'For a slab simulation, 2 dimentions have to fill box ' + \
                                       'n_cells_high=%s n_cells=%i' % (n_cells_high, n_cells)

        # Dimensions in Mpc/h of high resolution grid.
        high_res_boxsize = [cell_length * float(n_cells_high[0]),
                            cell_length * float(n_cells_high[1]),
                            cell_length * float(n_cells_high[2])]
        max_boxsize = np.max(high_res_boxsize)
        min_boxsize = np.min(high_res_boxsize)
        high_res_volume = high_res_boxsize[0] * high_res_boxsize[1] * high_res_boxsize[2]
        if comm_rank == 0:
            print('Length of high resolution grid [%.4f,%.4f,%.4f] Mpc/h in %i cells' \
                  % (high_res_boxsize[0], high_res_boxsize[1], high_res_boxsize[2],
                     n_cells_high[0] * n_cells_high[1] * n_cells_high[2]))
        if n_cells_high[0] * n_cells_high[1] * n_cells_high[2] > 2 ** 32 / 2.:
            raise Exception("Total number of high res cells too big")

        # Generate high resolution grid.
        if comm_rank == 0:
            print('Generating high resolution grid [%i %i %i]...' % \
                  (n_cells_high[0], n_cells_high[1], n_cells_high[2]))
        this_num_cells = (n_cells_high[0] * n_cells_high[1] * n_cells_high[2]) // comm_size
        if comm_rank < (n_cells_high[0] * n_cells_high[1] * n_cells_high[2]) % comm_size:
            this_num_cells += 1
        offsets, cell_nos = \
            get_grid(n_cells_high[0], n_cells_high[1], n_cells_high[2], comm_rank,
                     comm_size, this_num_cells)
        assert len(cell_nos) == this_num_cells, \
            'Error creating cells %i != %i' % (len(cell_nos), this_num_cells)
        assert comm.allreduce(len(cell_nos)) == n_cells_high[0] * n_cells_high[1] * n_cells_high[2], \
            'Error creating cells 2.'

        # Initiate the high resolution grid.
        cell_types_hi = self.init_high_res(offsets, cell_nos, n_cells_high[0])  # NOT RIGHT FOR SLAB

        # Total memory of high res grid.
        self.size_of_HR_grid_arrays = sys.getsizeof(cell_types_hi) + sys.getsizeof(offsets) + \
                                      sys.getsizeof(cell_nos)

        # Find total number of particles that will be stored on this core.
        side = np.true_divide(max_cells, n_cells)
        n_tot_hi, n_tot_lo = self.find_ntot(self.glass_num, cell_types_hi, side, max_cells,
                                            min_boxsize, max_boxsize)
        ntot = n_tot_hi + n_tot_lo
        all_ntot = comm.reduce(ntot)
        all_ntot_lo = comm.reduce(n_tot_lo)
        if comm_rank == 0:
            if self.is_zoom:
                if self.mask_file is not None:
                    n_particles_target = (self.n_particles / self.box_size ** 3.) \
                                         * self.mask_high_res_volume
                print('--- Total number of glass particles %i (%.2f cubed, %.2f percent)' % \
                      (self.n_tot_glass_part, self.n_tot_glass_part ** (1 / 3.), np.true_divide(
                          self.n_tot_glass_part, all_ntot)))
                print('--- Total number of grid particles %i (%.2f cubed, %.2f percent)' % \
                      (self.n_tot_grid_part, self.n_tot_grid_part ** (1 / 3.), np.true_divide(
                          self.n_tot_grid_part, all_ntot)))
                print('--- Total number of outer particles %i (%.2f cubed, %.2f percent)' % \
                      (all_ntot_lo, all_ntot_lo ** (1 / 3.), np.true_divide(
                          all_ntot_lo, all_ntot)))
                if self.mask_file is not None:
                    print('--- Target number of ps %i (%.2f cubed), made %.2f times as many.' % \
                          (n_particles_target, n_particles_target ** (1 / 3.),
                           np.true_divide(all_ntot, n_particles_target)))
                if self.is_slab:
                    high_res_n_eff = self.n_particles
                    high_res_L = self.box_size
                else:
                    high_res_L = self.ic_region_buffer_frac * max_boxsize
                    assert high_res_L < self.box_size, 'Zoom buffer region too big'
                    high_res_n_eff = int(self.n_particles * (high_res_L ** 3. / self.box_size ** 3.))
                print('--- HRgrid c=%s' % self.coords)
                print('--- HRgrid L_box=%.2f Mpc/h, L_grid=%.2f Mpc/h n_eff=%i (%.2f cub,2.5x=%.2f)' \
                      % (self.box_size, high_res_L, high_res_n_eff,
                         high_res_n_eff ** (1 / 3.), 2.5 * high_res_n_eff ** (1 / 3.)))
            print('--- Total number of particles %i (%.2f cubed)' % \
                  (all_ntot, all_ntot ** (1 / 3.)))
            print('--- Total memory per rank HR grid=%.6f Gb, total of particles=%.6f Gb' % \
                  (self.size_of_HR_grid_arrays / 1024. / 1024. / 1024.,
                   (4 * all_ntot * 8. / 1024. / 1024. / 1024.)))
            print('--- Num ranks needed for < 400**3 per rank = %.2f' % \
                  (np.true_divide(all_ntot, 400 ** 3.)))
        if self.only_calc_ntot:
            sys.exit()

        # Initiate arrays.
        coords_x = np.empty(ntot, dtype='f8')
        coords_y = np.empty(ntot, dtype='f8')
        coords_z = np.empty(ntot, dtype='f8')
        masses = np.empty(ntot, dtype='f8')

        # Load all the glass files we are going to need.
        glass = {}
        if self.grid_also_glass:
            for this_glass_no in self.cell_info['num_particles_per_cell']:
                if this_glass_no not in glass.keys():
                    glass[this_glass_no] = self.load_glass_file(this_glass_no)
        else:
            glass[self.glass_num] = self.load_glass_file(self.glass_num)

        # Populate high resolution grid with particles.
        self.populate_levels(cell_length, cell_types_hi, offsets, glass,
                             n_tot_hi, coords_x, coords_y, coords_z, masses)

        # Rescale masses and coordinates of high res particles and check COM.
        coords_x[:n_tot_hi] = self.rescale(coords_x[:n_tot_hi], -max_cells / 2., max_cells / 2.,
                                           -max_boxsize / 2, max_boxsize / 2.) / self.box_size  # -0.5 > +0.5.
        coords_y[:n_tot_hi] = self.rescale(coords_y[:n_tot_hi], -max_cells / 2., max_cells / 2.,
                                           -max_boxsize / 2, max_boxsize / 2.) / self.box_size  # -0.5 > +0.5.
        coords_z[:n_tot_hi] = self.rescale(coords_z[:n_tot_hi], -max_cells / 2., max_cells / 2.,
                                           -max_boxsize / 2, max_boxsize / 2.) / self.box_size  # -0.5 > +0.5.
        assert np.all(np.abs(coords_x[:n_tot_hi]) < 0.5), 'High res coords error x'
        assert np.all(np.abs(coords_y[:n_tot_hi]) < 0.5), 'High res coords error y'
        assert np.all(np.abs(coords_z[:n_tot_hi]) < 0.5), 'High res coords error z'
        tot_hr_mass = comm.allreduce(np.sum(masses[:n_tot_hi]))
        assert np.abs(tot_hr_mass - (high_res_volume / self.box_size ** 3.)) \
               <= 1e-6, 'Error high res masses %.8f' \
                        % (np.abs(tot_hr_mass - (high_res_volume / self.box_size ** 3.)))
        com_x, com_y, com_z = self.com(coords_x[:n_tot_hi], coords_y[:n_tot_hi],
                                       coords_z[:n_tot_hi], masses[:n_tot_hi])
        if comm_rank == 0:
            print('CoM for high res grid particles [%.2g %.2g %.2g]' % (com_x / tot_hr_mass,
                                                                        com_y / tot_hr_mass, com_z / tot_hr_mass))
        comm.barrier()

        # Generate outer particles of low res grid with growing skins.
        if self.is_zoom:
            if comm_rank == 0: print('\n------Outer low res skins ------')
            assert min_cells < n_cells, 'Cant zoom if the high res region is the whole box!'
            if self.is_slab:
                if comm_rank == 0:
                    print('Putting low res particles around slab of width %.2f Mpc/h' % \
                          min_boxsize)
                if n_tot_lo > 0:
                    get_layered_particles_slab(min_boxsize, self.box_size,
                                               self.nq_info['starting_nq'], self.nq_info['nlev_slab'],
                                               self.nq_info['dv_slab'], comm_rank, comm_size, n_tot_lo, n_tot_hi,
                                               coords_x, coords_y, coords_z, masses, self.nq_info['nq_reduce'],
                                               self.nq_info['extra'])
            else:
                if n_tot_lo > 0:
                    get_layered_particles(side, self.nq_info['nq'], comm_rank,
                                          comm_size, n_tot_lo, n_tot_hi, self.nq_info['extra'],
                                          self.nq_info['total_volume'], coords_x, coords_y, coords_z, masses)

            comm.barrier()
            if n_tot_lo > 0:
                print('[%i: Outer particles] Generated %i (%.2f cubed) MinM = %.2f MaxM = %.2f' % \
                      (comm_rank, n_tot_lo, n_tot_lo ** (1 / 3.), np.log10(np.min(masses[n_tot_hi:])),
                       np.log10(np.max(masses[n_tot_hi:]))))

                max_lr = np.max(masses[n_tot_hi:])
                min_lr = np.min(masses[n_tot_hi:])
                num_lr = len(masses[n_tot_hi:])
            else:
                max_lr = 1e-20
                min_lr = 1e20
                num_lr = 0

            max_lr = comm.allreduce(max_lr, op=MPI.MAX)
            min_lr = comm.allreduce(min_lr, op=MPI.MIN)
            num_lr = comm.allreduce(num_lr)

            if comm_rank == 0:
                print('Total %i particles in low res region (%.2f cubed) MinM = %.2f MaxM = %.2f' % \
                      (num_lr, num_lr ** (1 / 3.), np.log10(min_lr), np.log10(max_lr)))
            self.lr_mass_cut = min_lr

            # Checks.
            if n_tot_lo > 0:
                assert np.all(np.abs(coords_x[n_tot_hi:]) <= 0.5), 'Low res coords x error'
                assert np.all(np.abs(coords_y[n_tot_hi:]) <= 0.5), 'Low res coords y error'
                assert np.all(np.abs(coords_z[n_tot_hi:]) <= 0.5), 'Low res coords z error'
            final_tot_mass = comm.allreduce(np.sum(masses))
            tmp_tol = np.abs(1 - final_tot_mass)
            assert tmp_tol <= 1e-5, 'Final mass error %.8f != 0.0' % tmp_tol
            if tmp_tol > 1e-6 and comm_rank == 0:
                print("***Warming*** total final mass tol is %.8f" % tmp_tol)
            com_x, com_y, com_z = self.com(coords_x, coords_y, coords_z, masses)
            if comm_rank == 0:
                print('CoM for all particles [%.2g %.2g %.2g]' % (com_x / final_tot_mass,
                                                                  com_y / final_tot_mass, com_z / final_tot_mass))

            final_tot_num = comm.allreduce(len(masses))
            assert final_tot_num == comm.allreduce(ntot), 'Final array length error'
            if comm_rank == 0:
                print('Created %i (%.2f cubed) total particles.' \
                      % (final_tot_num, final_tot_num ** (1 / 3.)))

        # Wrap coords to chosen center.
        wrap_coords = self.rescale(self.coords, 0, self.box_size, 0, 1.0)
        if comm_rank == 0: print('Lagrangian COM of high res region = %s' % self.coords)
        coords_x = np.mod(coords_x + wrap_coords[0] + 1., 1.0)
        coords_y = np.mod(coords_y + wrap_coords[1] + 1., 1.0)
        coords_z = np.mod(coords_z + wrap_coords[2] + 1., 1.0)

        # Check coords and masses.
        assert np.all(coords_x > 0) and np.all(coords_x < 1.0), 'Coords x wrap error'
        assert np.all(coords_y > 0) and np.all(coords_y < 1.0), 'Coords y wrap error'
        assert np.all(coords_z > 0) and np.all(coords_z < 1.0), 'Coords z wrap error'
        assert np.all(masses > 0.0) and np.all(masses < 1.0), 'Mass number error'

        # Some boundary checks (Need to think about these, esp with 'extra' implementation).
        # if self.is_zoom:
        #    # Check the first PT3 particles are the right distance from the outer PT2 particles.
        #    idx = np.logical_and(masses > self.max_grid_mass,
        #            np.logical_and(coords_y > 0.5 - side/4., 
        #                np.logical_and(coords_y < 0.5 + side/4.,
        #                    np.logical_and(coords_z > 0.5 - side/4,
        #                        np.logical_and(coords_z < 0.5 + side/4., coords_x > 0.5)))))
        #    max_x =  comm.allreduce(np.min(coords_x[idx]), op=MPI.MIN)
        #    idx = np.where(coords_x == max_x)
        #    check_y = comm.gather(coords_y[idx])
        #    if comm_rank == 0:
        #        check_y = np.concatenate(check_y)
        #        print 'Distance from PT2 -> PT3 = %.8f, dy = %.8f (dx/dy = %.8f)'%\
        #            ((max_x - (side/2.+0.5)) * 2., np.diff(np.unique(check_y))[0],
        #            np.true_divide((max_x - (side/2.+0.5)) * 2., np.diff(np.unique(check_y))[0]))

        #    # Check opposite boundary particles are the right distance appart.
        #    max_x = comm.allreduce(np.max(coords_x), op=MPI.MAX)
        #    min_x = comm.allreduce(np.min(coords_x), op=MPI.MIN)
        #    idx = np.where(coords_x == max_x)
        #    check_y = comm.gather(coords_y[idx])
        #    if comm_rank == 0:
        #        check_y = np.concatenate(check_y)
        #        print 'Distance over boundary = %.8f, distance between grid in y %.8f (dx/dy = %.8f)'%\
        #                (1-max_x + min_x, np.diff(np.unique(check_y))[0],
        #                np.true_divide((1-max_x) + min_x, np.diff(np.unique(check_y))[0]))

        numtot = comm.allreduce(len(masses))
        if comm_rank == 0:
            self.save_param_files(max_boxsize)
            self.save_submit_files(max_boxsize)
        comm.barrier()

        # Save.
        if self.save_data: self.save(coords_x, coords_y, coords_z, masses)

    def save_param_files(self, max_boxsize):
        """ Create ic and gadget parameter files and submit scripts. """

        # Compute mass cut offs between particle types.
        hr_cut = 0.0
        lr_cut = 0.0

        if self.is_zoom:
            if self.n_species >= 2:
                hr_cut = np.log10(self.glass_particle_mass) + 0.01
                print('log10 mass cut from parttype 1 --> 2 = %.2f' % (hr_cut))
            if self.n_species == 3:
                lr_cut = np.log10(self.max_grid_mass) + 0.01
                print('log10 mass cut from parttype 2 --> 3 = %.2f' % (lr_cut))

            if self.is_slab:
                high_res_n_eff = self.n_particles
                high_res_L = self.box_size

            else:
                high_res_L = self.ic_region_buffer_frac * max_boxsize
                assert high_res_L < self.box_size, 'Zoom buffer region too big'
                high_res_n_eff = int(self.n_particles * (high_res_L ** 3. / self.box_size ** 3.))
            print('High res grid coords=%s, L_box=%.2f Mpc/h, L_grid=%.2f Mpc/h n_eff=%i (%.2f cubed)' \
                  % (self.coords, self.box_size, high_res_L, high_res_n_eff,
                     high_res_n_eff ** (1 / 3.)))
        else:
            high_res_n_eff = 0
            high_res_L = 0.0

        # Make ICs param file.
        i_z = 1 if self.is_zoom else 0
        make_param_file_ics(self.ic_dir, self.f_name, self.n_species, hr_cut, lr_cut,
                            self.box_size, self.starting_z, self.n_particles, self.coords[0],
                            self.coords[1], self.coords[2], high_res_L, high_res_n_eff, i_z,
                            self.panphasian_descriptor, self.constraint_phase_descriptor,
                            self.constraint_phase_descriptor_path, self.constraint_phase_descriptor_levels,
                            self.constraint_phase_descriptor2, self.constraint_phase_descriptor_path2,
                            self.constraint_phase_descriptor_levels2,
                            self.ndim_fft_start, self.Omega0, self.OmegaLambda, self.OmegaBaryon,
                            self.HubbleParam, self.Sigma8, self.is_slab, self.use_ph_ids,
                            self.multigrid_ics, self.linear_ps, self.nbit)
        print('\n------ Saving ------')
        print('Saved ics param file.')

        eps_dm, eps_baryon, eps_dm_physical, eps_baryon_physical = self.compute_softning()
        if self.do_gadget4:
            raise Exception("Think about DMO for gadget4")
            # Make GADGET4 param file.
            # make_param_file_gadget4(self.gadget4_dir, self.f_name, self.box_size,
            #        self.starting_z, self.finishing_z, self.Omega0, self.OmegaLambda,
            #        self.OmegaBaryon, self.HubbleParam, s_high)

            ## Makde GADGET4 submit file.
            # make_submit_file_gadget4(self.gadget4_dir, self.f_name)

        if self.do_gadget:
            raise Exception("Think about DMO for gadget")
            # Make GADGET param file.
            # make_param_file_gadget(self.gadget_dir, self.f_name, self.box_size,
            #    s_high, s_low, s_low_low, self.Omega0, self.OmegaLambda,
            #    self.OmegaBaryon, self.HubbleParam, self.dm_only,
            #    self.starting_z, self.finishing_z, self.high_output)
            # print 'Saved gadget param file.'

        if self.do_swift:
            # Make swift param file (remember no h's for swift).
            make_param_file_swift(self.swift_dir, self.Omega0, self.OmegaLambda,
                                  self.OmegaBaryon, self.HubbleParam, self.starting_z, self.finishing_z,
                                  self.f_name, self.is_zoom, self.template_set,
                                  self.softening_ratio_background, eps_dm, eps_baryon, eps_dm_physical,
                                  eps_baryon_physical)
            print('Saved swift param file.')

    def save_submit_files(self, max_boxsize):

        """ Generate submit files. """
        # Make ICs submit file.
        make_submit_file_ics(self.ic_dir, self.f_name,
                             self.num_hours_ic_gen, self.n_cores_ic_gen)
        print('Saved ics submit file.')

        if self.do_gadget:
            make_submit_file_gadget(self.gadget_dir, self.n_cores_gadget, self.f_name,
                                    self.n_species, self.gadget_exec)
            print('Saved gadget submit file.')

        if self.do_swift:
            # Make swift submit files.
            make_submit_file_swift(self.swift_dir, self.f_name,
                                   self.n_nodes_swift, self.num_hours_swift, self.template_set)
            print('Saved swift submit file.')

    def compute_softning(self, verbose=False):
        """ Compute softning legnths. """

        if self.dm_only:
            comoving_ratio = 1 / 20.  # = 0.050
            physical_ratio = 1 / 45.  # = 0.022
        else:
            comoving_ratio = 1 / 20.  # = 0.050
            physical_ratio = 1 / 45.  # = 0.022

        N = self.n_particles ** (1 / 3.)
        mean_inter = self.box_size / N

        # DM 
        eps_dm = mean_inter * comoving_ratio
        eps_dm_physical = mean_inter * physical_ratio

        # Baryons
        if self.dm_only:
            eps_baryon = 0.0
            eps_baryon_physical = 0.0
        else:
            fac = ((self.Omega0 - self.OmegaBaryon) / self.OmegaBaryon) ** (1. / 3)
            eps_baryon = eps_dm / fac
            eps_baryon_physical = eps_dm_physical / fac

        if comm_rank == 0 and verbose:
            print('Comoving Softenings: DM=%.6f Baryons=%.6f Mpc/h' % (
                eps_dm, eps_baryon))
            print('Max phys Softenings: DM=%.6f Baryons=%.6f Mpc/h' % (
                eps_dm_physical, eps_baryon_physical))
            print('Comoving Softenings: DM=%.6f Baryons=%.6f Mpc' % (
                eps_dm / self.HubbleParam, eps_baryon / self.HubbleParam))
            print('Max phys Softenings: DM=%.6f Baryons=%.6f Mpc' % (
                eps_dm_physical / self.HubbleParam,
                eps_baryon_physical / self.HubbleParam))

        return eps_dm, eps_baryon, eps_dm_physical, eps_baryon_physical

    def com(self, coords_x, coords_y, coords_z, masses):
        """ Compute center of mass of a list of coords. """
        com_x = np.sum(coords_x * masses)
        com_y = np.sum(coords_y * masses)
        com_z = np.sum(coords_z * masses)

        return comm.reduce(com_x), comm.reduce(com_y), comm.reduce(com_z)

    def generate_uniform_grid(self, n_particles):
        """ For low res regions just making a uniform grid of particles. """
        if n_particles == 1:
            coords = np.ones((1, 3), dtype='f8') * 0.5
        else:
            L = int(np.rint(n_particles ** (1 / 3.)))
            coords = np.zeros((n_particles, 3), dtype='f8')
            count = 0
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        coords[count][0] = (i + 0.5) / L
                        coords[count][1] = (j + 0.5) / L
                        coords[count][2] = (k + 0.5) / L
                        count += 1

        assert np.all(coords >= 0.0) and np.all(coords <= 1.0), 'Error uniform grid'
        return coords

    def save(self, coords_x, coords_y, coords_z, masses):
        ntot = comm.allreduce(len(masses))

        # Randomise arrays.
        if self.randomize:
            if comm_rank == 0: print('Randomizing arrays...')
            n_tot_particles = len(masses)
            idx = np.random.permutation(len(masses))
            assert np.all(idx >= 0) and np.all(idx < n_tot_particles), 'Error making random index'
            coords_x = coords_x[idx]
            coords_y = coords_y[idx]
            coords_z = coords_z[idx]
            masses = masses[idx]

        # Load balance.
        if comm_size > 1:
            ndesired = np.zeros(comm_size, dtype=int)
            ndesired[:] = ntot / comm_size
            ndesired[-1] += (ntot - sum(ndesired))
            if comm_rank == 0:
                tmp_num_per_file = ndesired[0] ** (1 / 3.)
                print('Load balancing %i particles on %i ranks (%.2f**3 per file)...' \
                      % (ntot, comm_size, tmp_num_per_file))
                if tmp_num_per_file > 400.:
                    print("***WARNING*** more than 400**3 per file***")

            masses = repartition(masses, ndesired, comm, comm_rank, comm_size)
            coords_x = repartition(coords_x, ndesired, comm, comm_rank, comm_size)
            coords_y = repartition(coords_y, ndesired, comm, comm_rank, comm_size)
            coords_z = repartition(coords_z, ndesired, comm, comm_rank, comm_size)
            if comm_rank == 0: print('Done load balancing.')

        comm.barrier()
        assert len(masses) == len(coords_x) == len(coords_y) == len(coords_z), 'Array length error'

        """ Save particle load to HDF5 file. """
        save_dir = '%s/%s/' % (self.pl_dir, self.f_name)
        save_dir_hdf = save_dir + 'hdf5/'
        save_dir_bin = save_dir + 'fbinary/'
        if comm_rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(save_dir_hdf) and self.save_hdf5:
                os.makedirs(save_dir_hdf)
            if not os.path.exists(save_dir_bin):
                os.makedirs(save_dir_bin)
        comm.barrier()

        # Make sure not to save more than max_save at a time.
        max_save = 50
        lo_ranks = np.arange(0, comm_size + max_save, max_save)[:-1]
        hi_ranks = np.arange(0, comm_size + max_save, max_save)[1:]
        for lo, hi in zip(lo_ranks, hi_ranks):
            if comm_rank >= lo and comm_rank < hi:

                if self.save_hdf5:
                    if comm_rank == 0:
                        print('Saving HDF5 files...')
                    f = h5py.File(save_dir_hdf + 'PL.%d.hdf5' % comm_rank, 'w')
                    f.create_dataset('Coordinates_x', data=np.array(coords_x, dtype='f8'))
                    f.create_dataset('Coordinates_y', data=np.array(coords_y, dtype='f8'))
                    f.create_dataset('Coordinates_z', data=np.array(coords_z, dtype='f8'))
                    f.create_dataset('Masses', data=np.array(masses, dtype='f4'))
                    g = f.create_group('Header')
                    g.attrs.create('nlist', len(masses))
                    g.attrs.create('itot', ntot)
                    g.attrs.create('nj', comm_rank)
                    g.attrs.create('nfile', comm_size)
                    f.close()

                # Save to fortran binary.
                f = open('%s/PL.%d' % (save_dir_bin, comm_rank), 'w')
                dt = [('nlist', np.int32), ('itot', np.int64), ('nj', np.int32),
                      ('nfile', np.int32), ('ibuf', np.int32, 7)]
                head = np.empty(1, dtype=dt)
                head['nlist'] = len(masses)
                head['itot'] = ntot
                head['nj'] = comm_rank
                head['nfile'] = comm_size
                head['ibuf'] = [0, 0, 0, 0, 0, 0, 0]

                self.write_fortran(head, f)
                self.write_fortran(np.array(coords_x, dtype=np.float64), f)
                self.write_fortran(np.array(coords_y, dtype=np.float64), f)
                self.write_fortran(np.array(coords_z, dtype=np.float64), f)
                self.write_fortran(np.array(masses, dtype=np.float32), f)
                f.close()

                print('[%i] Saved %i/%i particles...' % (comm_rank, len(masses), ntot))
            comm.barrier()

    def write_fortran(self, s, f):
        s = np.array(s, order='F')
        np.array([s.nbytes], dtype=np.int32).tofile(f)
        s.tofile(f)
        np.array([s.nbytes], dtype=np.int32).tofile(f)


if __name__ == '__main__':
    only_calc_ntot = False
    if len(sys.argv) > 2:
        only_calc_ntot = True if int(sys.argv[2]) == 1 else False

    ParticleLoad(sys.argv[1], only_calc_ntot=only_calc_ntot)
