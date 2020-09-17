import sys
import yaml
import h5py
from typing import List, Tuple
from warnings import warn
import numpy as np
from scipy.spatial import distance
from scipy import ndimage
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from read_swift import read_swift
from read_eagle import EagleSnapshot
from peano import peano_hilbert_key_inverses

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size

# try:
#     plt.style.use("../../mnras.mplstyle")
# except:
#     pass

output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/ics/masks"


class MakeMask:

    def __init__(self, param_file):

        self.read_param_file(param_file)
        self.make_mask()

    def read_param_file(self, param_file):
        """ Read parameters from YAML file. """
        if comm_rank == 0:
            params = yaml.load(open(param_file))

            # Defaults.
            self.params = {}
            self.params['GN'] = None
            self.params['data_type'] = 'swift'
            self.params['min_num_per_cell'] = 3
            self.params['mpc_cell_size'] = 3.  # Cell size in Mpc/h
            self.params['select_from_vr'] = False

            required_params = [
                'select_from_vr',
                'fname',
                'snap_file',
                'bits',
                'shape',
                'data_type',
                'divide_ids_by_two'
            ]

            for att in required_params:
                assert att in params.keys(), 'Need to have %s as a param' % att

            # Run checks for automatic and manual group selection
            if params['select_from_vr']:
                assert 'GN' in params.keys(), 'Need to provide a Group-Number for the group'
                assert 'vr_file' in params.keys(), 'Need to provide a sub_file'
                assert 'highres_radius_r200' in params.keys() or 'highres_radius_r500' in params.keys(), \
                    'Need to provide a radius for the high-resolution region in either R200crit ot R500crit units.'

            else:
                assert 'shape' in params.keys(), 'Need to provide a shape of region.'
                assert 'coords' in params.keys(), 'Need to provide coords of region.'
                if params['shape'] == 'cuboid' or params['shape'] == 'slab':
                    assert 'dim' in params.keys(), 'Need to provide dimensions of region.'
                elif params['shape'] == 'sphere':
                    assert 'radius' in params.keys(), 'Need to provide radius of sphere.'

            for att in params.keys():
                self.params[att] = params[att]
        else:
            self.params = None

        self.params = comm.bcast(self.params)

        # Find the group we want to re-simulate (if selected)
        if self.params['select_from_vr']:
            self.params['coords'], self.params['radius'] = self.find_group()
            self.params['shape'] = 'sphere'

        self.params['coords'] = np.array(self.params['coords'], dtype='f8')
        if 'dim' in self.params.keys():
            self.params['dim'] = np.array(self.params['dim'])

    def find_group(self) -> Tuple[List[float], float]:

        # Read in halo properties
        with h5py.File(self.params['vr_file'], 'r') as vr_file:
            structType = vr_file['/Structuretype'][:]
            field_halos = np.where(structType == 10)[0]
            R200c = vr_file['/R_200crit'][field_halos][self.params['GN']]

            try:
                R500c = vr_file['/R_500crit'][field_halos][self.params['GN']]
            except KeyError as error:
                if comm_rank == 0:
                    print(error)
                    print("If using highres_radius_r500, the selection will use R_200crit instead.")
                    warn("The high-resolution radius is now set to R_200crit * highres_radius_r500 / 2.", RuntimeWarning)
                R500c = R200c / 2

            xPotMin = vr_file['/Xcminpot'][field_halos][self.params['GN']]
            yPotMin = vr_file['/Ycminpot'][field_halos][self.params['GN']]
            zPotMin = vr_file['/Zcminpot'][field_halos][self.params['GN']]

        is_r200 = 'highres_radius_r200' in self.params.keys()
        is_r500 = 'highres_radius_r500' in self.params.keys()

        # If no radius is selected, use the default R200
        if is_r200 and is_r500:
            if comm_rank == 0:
                warn("Both highres_radius_r200 and highres_radius_r500 were entered. Overriding highres_radius_r500.")
            radius = R200c * self.params['highres_radius_r200']
        elif not is_r200 and is_r500:
            radius = R500c * self.params['highres_radius_r500']
        elif is_r200 and not is_r500:
            radius = R200c * self.params['highres_radius_r200']
        else:
            raise ValueError("Neither highres_radius_r200 nor highres_radius_r500 were entered.")

        if comm_rank == 0:
            print(
                "Velociraptor search results:\n"
                f"- Run name: {self.params['fname']}\tGroupNumber: {self.params['GN']}\n"
                f"- Coordinate centre: ", ([xPotMin, yPotMin, zPotMin]), "\n"
                f"- High-res radius: {radius}\n"
                f"- R200_crit: {R200c}\n"
                f"- R500_crit: {R500c}\n"
            )

        return [xPotMin, yPotMin, zPotMin], radius

    def compute_ic_positions(self, ids) -> np.ndarray:
        """ Compute the positions at ICs. """
        print('[Rank %i] Computing initial positions of dark matter particles...' % comm_rank)
        X, Y, Z = peano_hilbert_key_inverses(ids, self.params['bits'])
        ic_coords = np.vstack((X, Y, Z)).T
        assert 0 <= np.all(ic_coords) < 2 ** self.params['bits'], 'Initial coords out of range'
        return np.array(ic_coords, dtype='f8')

    def load_particles(self):
        """ Load particles from chosen snapshot."""

        if self.params['data_type'].lower() == 'gadget':
            snap = EagleSnapshot(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = 1.0
            self.params['length_unit'] = 'Mph/h'
        elif self.params['data_type'].lower() == 'swift':
            snap = read_swift(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = float(snap.COSMOLOGY['h'])
            self.params['length_unit'] = 'Mpc'

        # A sphere with radius R.
        if self.params['shape'] == 'sphere':
            region = [self.params['coords'][0] - self.params['radius'],
                      self.params['coords'][0] + self.params['radius'],
                      self.params['coords'][1] - self.params['radius'],
                      self.params['coords'][1] + self.params['radius'],
                      self.params['coords'][2] - self.params['radius'],
                      self.params['coords'][2] + self.params['radius']]
        # A cuboid with sides x,y,z.
        elif self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            region = [self.params['coords'][0] - self.params['dim'][0] / 2.,
                      self.params['coords'][0] + self.params['dim'][0] / 2.,
                      self.params['coords'][1] - self.params['dim'][1] / 2.,
                      self.params['coords'][1] + self.params['dim'][1] / 2.,
                      self.params['coords'][2] - self.params['dim'][2] / 2.,
                      self.params['coords'][2] + self.params['dim'][2] / 2.]
        if comm_rank == 0:
            print('Loading region...\n', region)
        if self.params['data_type'].lower() == 'gadget':
            snap.select_region(*region)
            snap.split_selection(comm_rank, comm_size)
        elif self.params['data_type'].lower() == 'swift':
            snap.select_region(1, *region)
            snap.split_selection(comm)

        # Load DM particle IDs.
        if comm_rank == 0: print('Loading particle data ...')
        ids = snap.read_dataset(1, 'ParticleIDs')
        coords = snap.read_dataset(1, 'Coordinates')
        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles.')

        # Wrap coordinates.
        coords = np.mod(coords - self.params['coords'] + 0.5 * self.params['bs'],
                        self.params['bs']) + self.params['coords'] - 0.5 * self.params['bs']

        # Clip to desired shape.
        if self.params['shape'] == 'sphere':
            if comm_rank == 0:
                print('Clipping to sphere around %s, radius %.4f %s' % (
                    self.params['coords'],
                    self.params['radius'],
                    self.params['length_unit']
                ))
            dists = distance.cdist(coords, self.params['coords'].reshape(1, 3), metric='euclidean').reshape(
                len(coords), )
            mask = np.where(dists <= self.params['radius'])

        elif self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            if comm_rank == 0:
                print('Clipping to %s x=%.2f %s, y=%.2f %s, z=%.2f %s around %s %s' \
                      % (self.params['shape'], self.params['dim'][0], self.params['length_unit'],
                         self.params['dim'][1], self.params['length_unit'],
                         self.params['dim'][2], self.params['length_unit'],
                         self.params['coords'], self.params['length_unit']))

            mask = np.where(
                np.logical_and(coords[:, 0] >= (self.params['coords'][0] - self.params['dim'][0] / 2.),
                np.logical_and(coords[:, 0] <= (self.params['coords'][0] + self.params['dim'][0] / 2.),
                np.logical_and(coords[:, 1] >= (self.params['coords'][1] - self.params['dim'][1] / 2.),
                np.logical_and(coords[:, 1] <= (self.params['coords'][1] + self.params['dim'][1] / 2.),
                np.logical_and(coords[:, 2] >= (self.params['coords'][2] -self.params['dim'][2] / 2.),
                coords[:, 2] <= (self.params['coords'][2] +self.params['dim'][2] / 2.)))))))

        ids = ids[mask]
        print(f'[Rank {comm_rank}] Clipped to {len(ids)} dark matter particles.')

        # Put back into original IDs.
        if self.params['divide_ids_by_two']:
            ids /= 2

        return ids, coords

    def convert_to_inverse_h(self, coords: np.ndarray) -> np.ndarray:
        h = self.params['h_factor']
        keys = self.params.keys()
        if 'radius' in keys:
            self.params['radius'] *= h
        if 'dim' in keys:
            self.params['dim'] *= h
        if 'coords' in keys:
            self.params['coords'] *= h
        if 'bs' in keys:
            self.params['bs'] *= h
        return coords * h

    def make_mask(self):
        # Load particles.
        ids, coords = self.load_particles()
        if self.params['data_type'].lower() == 'swift':
            coords = self.convert_to_inverse_h(coords)

        # Find initial positions from IDs.
        ic_coords = self.compute_ic_positions(ids)

        # Rescale IC coords to 0-->boxsize.
        ic_coords *= np.true_divide(self.params['bs'], 2 ** self.params['bits'] - 1)
        ic_coords = np.mod(ic_coords - self.params['coords'] + 0.5 * self.params['bs'],
                           self.params['bs']) + self.params['coords'] - 0.5 * self.params['bs']

        # Find COM of the lagrangian region.
        count = 0
        last_com_coords = np.array([self.params['bs'] / 2., self.params['bs'] / 2., self.params['bs'] / 2.])

        while True:
            com_coords = self.get_com(ic_coords, self.params['bs'])
            if comm_rank == 0:
                print(f'COM iteration {count} c={com_coords} Mpc/h')
            ic_coords = np.mod(ic_coords - com_coords + 0.5 * self.params['bs'],
                               self.params['bs']) + com_coords - 0.5 * self.params['bs']
            if np.sum(np.abs(com_coords - last_com_coords)) <= 1e-6:
                break
            last_com_coords = com_coords
            count += 1
            if (count > 10) or (self.params['shape'] == 'slab'):
                break
        if comm_rank == 0:
            print('COM of lagrangian region %s Mpc/h\n\t(compared to coords %s Mpc/h)' \
                  % (com_coords, self.params['coords']))
        ic_coords -= com_coords

        # Compute outline
        num_bins = int(np.ceil(self.params['bs'] / (self.params['mpc_cell_size'])))
        bins = np.linspace(-self.params['bs'] / 2., self.params['bs'] / 2., num_bins)
        bin_width = bins[1] - bins[0]
        H, edges = np.histogramdd(ic_coords, bins=(bins, bins, bins))
        H = comm.allreduce(H)

        # Initialize binary mask
        bin_mask = np.zeros_like(H, dtype=np.bool)
        m = np.where(H >= self.params['min_num_per_cell'])
        bin_mask[m] = True

        # Fill holes and extrude the mask
        if comm_rank == 0:
            print("(1/3) [Topological extrusion] Scanning x-y plane...")
        for layer_id in range(bin_mask.shape[0]):
            bin_mask[layer_id, :, :] = ndimage.binary_dilation(bin_mask[layer_id, :, :], iterations=1).astype(np.bool)
            # bin_mask[layer_id, :, :] = ndimage.binary_closing(bin_mask[layer_id, :, :], iterations=1).astype(np.bool)

        if comm_rank == 0:
            print("(2/3) [Topological extrusion] Scanning y-z plane...")
        for layer_id in range(bin_mask.shape[1]):
            bin_mask[:, layer_id, :] = ndimage.binary_dilation(bin_mask[:, layer_id, :], iterations=1).astype(np.bool)
            # bin_mask[:, layer_id, :] = ndimage.binary_closing(bin_mask[:, layer_id, :], iterations=1).astype(np.bool)

        if comm_rank == 0:
            print("(3/3) [Topological extrusion] Scanning x-z plane...")
        for layer_id in range(bin_mask.shape[2]):
            bin_mask[:, :, layer_id] = ndimage.binary_dilation(bin_mask[:, :, layer_id], iterations=1).astype(np.bool)
            # bin_mask[:, :, layer_id] = ndimage.binary_closing(bin_mask[:, :, layer_id], iterations=1).astype(np.bool)


        # Computing bounding region
        m = np.where(bin_mask == True)
        lens = np.array([np.abs(np.min(edges[0][m[0]])),
                         np.max(edges[0][m[0]]) + bin_width,
                         np.abs(np.min(edges[1][m[1]])),
                         np.max(edges[1][m[1]]) + bin_width,
                         np.abs(np.min(edges[2][m[2]])),
                         np.max(edges[2][m[2]]) + bin_width])

        if comm_rank == 0:
            print(
                f"Encompassing dimensions:\n"
                f"\tx = {(lens[0] + lens[1]):.4f} Mpc/h\n"
                f"\ty = {(lens[2] + lens[3]):.4f} Mpc/h\n"
                f"\tz = {(lens[4] + lens[5]):.4f} Mpc/h"
            )

            tot_cells = len(H[0][m[0]]) + len(H[1][m[1]]) + len(H[2][m[2]])
            print(f'There are {tot_cells:d} total glass cells.')
        # Plot.
        self.plot(H, edges, bin_width, m, ic_coords, lens)

        # Save.
        if comm_rank == 0:
            self.save(H, edges, bin_width, m, lens, com_coords)

    def plot(self, H, edges, bin_width, m, ic_coords, lens):
        """ Plot the region outline. """
        axes_label = ['x', 'y', 'z']
        # Subsample.
        idx = np.random.permutation(len(ic_coords))
        if len(ic_coords) > 1e5:
            idx = np.random.permutation(len(ic_coords))
            plot_coords = ic_coords[idx][:100000]
        else:
            plot_coords = ic_coords[idx]

        plot_coords = comm.gather(plot_coords)

        if comm_rank == 0:
            plot_coords = np.vstack(plot_coords)
            fig, axarr = plt.subplots(1, 3, figsize=(10, 4))

            for count, (i, j) in enumerate(zip([0, 0, 1], [1, 2, 2])):
                axarr[count].set_aspect('equal')
                rect = patches.Rectangle(
                    (-lens[i * 2], -lens[j * 2]),
                    lens[i * 2 + 1] + lens[i * 2],
                    lens[j * 2 + 1] + lens[j * 2],
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                axarr[count].scatter(plot_coords[:, i], plot_coords[:, j], s=0.5, c='blue')
                axarr[count].add_patch(rect)
                axarr[count].set_xlim(-lens[i * 2], lens[i * 2 + 1])
                axarr[count].set_ylim(-lens[j * 2], lens[j * 2 + 1])

                axarr[count].scatter(
                    edges[i][m[i]] + bin_width / 2.,
                    edges[j][m[j]] + bin_width / 2.,
                    marker='^', color='red', s=3
                )

                if len(m[i]) < 10000:
                    for e_x, e_y in zip(edges[i][m[i]], edges[j][m[j]]):
                        rect = patches.Rectangle(
                            (e_x, e_y),
                            bin_width,
                            bin_width,
                            linewidth=0.5,
                            edgecolor='r',
                            facecolor='none'
                        )
                        axarr[count].add_patch(rect)

                axarr[count].set_xlabel(f"{axes_label[i]} [Mpc h$^{{-1}}$]")
                axarr[count].set_ylabel(f"{axes_label[j]} [Mpc h$^{{-1}}$]")

            plt.tight_layout(pad=0.3)
            fig.savefig(f"{output_directory}/{self.params['fname']:s}.png")
            plt.show()

    def save(self, H, edges, bin_width, m, lens, com_coords):
        # Save (everything needs to be saved in h inverse units, for the IC GEN).
        f = h5py.File(f"{output_directory}/{self.params['fname']:s}.hdf5", 'w')
        coords = np.c_[edges[0][m[0]] + bin_width / 2.,
                       edges[1][m[1]] + bin_width / 2.,
                       edges[2][m[2]] + bin_width / 2.]
        ds = f.create_dataset('Coordinates', data=np.array(coords, dtype='f8'))
        ds.attrs.create('xlen_lo', lens[0])
        ds.attrs.create('xlen_hi', lens[1])
        ds.attrs.create('ylen_lo', lens[2])
        ds.attrs.create('ylen_hi', lens[3])
        ds.attrs.create('zlen_lo', lens[4])
        ds.attrs.create('zlen_hi', lens[5])
        ds.attrs.create('coords', self.params['coords'])
        ds.attrs.create('com_coords', com_coords)
        ds.attrs.create('grid_cell_width', bin_width)
        if self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            ds.attrs.create('high_res_volume',
                            self.params['dim'][0] \
                            * self.params['dim'][1] \
                            * self.params['dim'][2])
        else:
            ds.attrs.create('high_res_volume', 4 / 3. * np.pi * self.params['radius'] ** 3.)
        f.close()
        print(f"Saved {output_directory}/{self.params['fname']:s}.hdf5")

    def get_com(self, ic_coords, bs):
        """ Find centre of mass for passed coordinates. """
        if self.params['shape'] == 'slab':
            com_x = bs / 2.
            com_y = bs / 2.
        else:
            com_x = comm.allreduce(np.sum(ic_coords[:, 0])) / comm.allreduce(len(ic_coords[:, 0]))
            com_y = comm.allreduce(np.sum(ic_coords[:, 1])) / comm.allreduce(len(ic_coords[:, 1]))
        com_z = comm.allreduce(np.sum(ic_coords[:, 2])) / comm.allreduce(len(ic_coords[:, 2]))
        return np.array([com_x, com_y, com_z])


if __name__ == '__main__':
    x = MakeMask(sys.argv[1])
