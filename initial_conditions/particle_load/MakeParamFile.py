import re
import os
import subprocess


# |------------------------------|
# | Make submit file for GADGET. |
# |------------------------------|

def make_submit_file_gadget(dir, ncores, fname, n_sp, structure_finding, exec_file):
    r = ['%i' % ncores, fname, fname, fname, exec_file]

    if not os.path.exists(dir + fname): os.makedirs(dir + fname)
    if not os.path.exists(dir + fname + '/out_files/'): os.makedirs(dir + fname + '/out_files/')
    if not os.path.exists(dir + fname + '/data/'): os.makedirs(dir + fname + '/data/')

    # GADGET submit file.
    with open('../templates/gadget_submit.sh', 'r') as f:
        data = f.read()

    with open('%s/%s/submit.sh' % (dir, fname), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))

    # GADGET resubmit file.
    with open('../templates/gadget_resubmit.sh', 'r') as f:
        data = f.read()

    with open('%s/%s/resubmit.sh' % (dir, fname), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))

    # GADGET auto resubmit file.
    with open('%s/%s/auto_resubmit.sh' % (dir, fname), 'w') as f:
        f.write('sbatch resubmit.sh')


# |------------------------------------|
# | Make submit file for IC generator. |
# |------------------------------------|

def make_submit_file_ics(dir, fname, num_hours, ncores):
    # First load template.
    with open('../templates/ic_gen/submit', 'r') as f:
        data = f.read()

    # Write new param file.
    r = ['%i' % ncores, fname, fname, '%i' % num_hours, fname]

    with open('%s/Submit_Files/%s.sh' % (dir, fname), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))


# |---------------------------------------|
# | Make parameter file for IC generator. |
# |---------------------------------------|

def make_param_file_ics(dir, fname, n_sp, sp1_2_div,
                        sp_2_3_div, boxsize, starting_z,
                        ntot, x, y, z, L, nhi, is_zoom, panphasian_descriptor,
                        constraint_phase_descriptor, constraint_phase_descriptor_path,
                        constraint_phase_descriptor_levels, constraint_phase_descriptor2,
                        constraint_phase_descriptor_path2,
                        constraint_phase_descriptor_levels2, ndim_fft_start, omega0, omegaL,
                        omegaB, h, sigma8, is_slab, use_ph_ids, multigrid_ics, linear_ps_file,
                        nbit):
    # Compute the minimum FFT grid that we need.
    ndim_fft = ndim_fft_start
    N = (nhi) ** (1. / 3) if is_zoom else (ntot) ** (1 / 3.)
    while float(ndim_fft) / float(N) < 2.5:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)

    # What are the phase descriptors?
    if constraint_phase_descriptor != '%dummy':
        if constraint_phase_descriptor2 != '%dummy':
            is_constraint = 2
        else:
            is_constraint = 1
    else:
        is_constraint = 0
    constraint_path = '%dummy' if constraint_phase_descriptor == '%dummy' else \
        "'%s'" % constraint_phase_descriptor_path
    constraint_path2 = '%dummy' if constraint_phase_descriptor2 == '%dummy' else \
        "'%s'" % constraint_phase_descriptor_path2

    # Is this a zoom simulation?
    if is_zoom:
        if is_slab:
            two_lpt = 1
            multigrid = 0
        else:
            two_lpt = 0 if multigrid_ics else 1
            multigrid = 1 if multigrid_ics else 0
    else:
        L = 0.0
        nhi = 0
        two_lpt = 1
        multigrid = 0

    # Use peano hilbert indexing?
    use_ph = 2 if use_ph_ids else 1

    # Replace values.
    r = [fname, '%i' % use_ph, '%i' % nbit, '%i' % n_sp, '%.2f' % sp1_2_div,
         '%.2f' % sp_2_3_div, '%i' % two_lpt,
         fname, fname, linear_ps_file,
         '%.8f' % boxsize, '%.8f' % omega0, '%.8f' % omegaL, '%.8f' % sigma8, '%.8f' % h, '%.8f' % starting_z,
         panphasian_descriptor, '%i' % is_constraint,
         constraint_phase_descriptor, constraint_path,
         constraint_phase_descriptor_levels, constraint_phase_descriptor2,
         constraint_path2, constraint_phase_descriptor_levels2,
         '%i' % ntot, '%i' % ndim_fft, '%i' % multigrid, '%.8f' % x, '%.8f' % y, '%.8f' % z, '%.8f' % L, '%i' % nhi]

    # First load template.
    with open('../templates/ic_gen/params.inp', 'r') as f:
        data = f.read()

    # Write new param file.
    save_dir = '%s/Param_Files/' % dir
    with open(save_dir + '/%s.inp' % fname, 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))


# |----------------------------------|
# | Make parameter file for GADGET4. |
# |----------------------------------|

def make_param_file_gadget4(dir, fname, boxsize, starting_z, finishing_z,
                            omega0, omegaL, omegaB, h, soft_high):
    # Make data dir.
    base_dir = '%s/%s/' % (dir, fname)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    data_dir = base_dir + 'output/'
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    # Starting and finishing scale factors.
    starting_a = 1. / (1 + starting_z)
    finishing_a = 1. / (1 + finishing_z)

    a = 1. / (1 + 2.8)
    r = [fname, '%.8f' % starting_a, '%.8f' % finishing_a, '%.8f' % boxsize,
         '%.8f' % omega0, '%.8f' % omegaL, '%.8f' % omegaB, '%.8f' % h,
         '%.8f' % soft_high, '%.8f' % (soft_high * a)]

    # First load template.
    with open('../templates/gadget4_params.txt', 'r') as f:
        data = f.read()

    # Write new param file.
    with open('%s/params.txt' % (base_dir), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))


# |-------------------------------|
# | Make submit file for Gadget4. |
# |-------------------------------|

def make_submit_file_gadget4(dir, fname):
    base_dir = '%s/%s/' % (dir, fname)
    assert os.path.exists(base_dir), 'G4 dir not exist'

    r = []

    with open('../templates/gadget4_submit.sh', 'r') as f:
        data = f.read()

    # Write new param file.
    with open('%s/%s/submit' % (dir, fname), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))


# |--------------------------------|
# | Make parameter file for SWIFT. |
# |--------------------------------|

def make_param_file_swift(dir, omega0, omegaL, omegaB, h, starting_z, finishing_z, fname,
                          structure_finding, is_zoom, template_set,
                          softening_ratio_background, eps_dm, eps_baryon, eps_dm_physical, eps_baryon_physical):
    # Make data dir.
    data_dir = dir + '%s/' % fname
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(data_dir + 'out_files/'): os.makedirs(data_dir + 'out_files/')
    if 'tabula_' in template_set.lower():
        if not os.path.exists(data_dir + 'los/'): os.makedirs(data_dir + 'los/')

    # Starting and finishing scale factors.
    starting_a = 1. / (1 + starting_z)
    finishing_a = 1. / (1 + finishing_z)

    # Replace values.
    if 'tabula_' in template_set.lower():
        r = ['%.5f' % h, '%.8f' % starting_a, '%.8f' % finishing_a, '%.8f' % omega0, '%.8f' % omegaL,
             '%.8f' % omegaB, fname, fname, '%.8f' % (eps_dm / h),
             '%.8f' % (eps_baryon / h), '%.3f' % (softening_ratio_background),
             '%.8f' % (eps_baryon_physical / h), '%.8f' % (eps_dm_physical / h), fname]

        subprocess.call("cp ../templates/swift/%s/select_output.yml %s" % \
                        (template_set, data_dir), shell=True)
    elif template_set.lower() == 'default':
        r = ['%.5f' % h, '%.8f' % starting_a, '%.8f' % finishing_a, '%.8f' % omega0, '%.8f' % omegaL,
             '%.8f' % omegaB, fname, '%.8f' % (eps_dm / h),
             '%.8f' % (eps_baryon / h), '%.3f' % (softening_ratio_background),
             '%.8f' % (eps_baryon_physical / h), '%.8f' % (eps_dm_physical / h), fname]
    elif template_set.lower() == 'sibelius':
        r = ['%.5f' % h, '%.8f' % starting_a, '%.8f' % finishing_a, '%.8f' % omega0, '%.8f' % omegaL,
             '%.8f' % omegaB, fname, '%.8f' % (eps_dm / h),
             '%.8f' % (eps_baryon / h), '%.3f' % (softening_ratio_background),
             '%.8f' % (eps_baryon_physical / h), '%.8f' % (eps_dm_physical / h), fname]
    elif template_set.lower() == 'dmo':
        r = ['%.5f' % h, '%.8f' % starting_a, '%.8f' % finishing_a, '%.8f' % omega0, '%.8f' % omegaL,
             '%.8f' % omegaB, fname, '%.8f' % (eps_dm / h),
             '%.8f' % (eps_baryon / h), '%.3f' % (softening_ratio_background),
             '%.8f' % (eps_baryon_physical / h), '%.8f' % (eps_dm_physical / h), fname]
    else:
        raise ValueError("Invalid template set")

    t_file = '../templates/swift/%s/params.yml' % template_set

    # Load the template.
    with open(t_file, 'r') as f:
        data = f.read()

    # Write new param file.
    with open('%s/params.yml' % data_dir, 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))


# |-----------------------------|
# | Make submit file for Swift. |
# |-----------------------------|

def make_submit_file_swift(dir, fname, structure_finding, n_nodes, num_hours, template_set):
    data_dir = dir + '%s/' % fname
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    extras = ''
    if structure_finding: extras = extras + '--velociraptor '
    if 'tabula_' in template_set.lower():
        extras = extras + '--line-of-sight '

    r = ['%i' % n_nodes, fname, '%i' % num_hours, extras]
    s_file = '../templates/swift/%s/submit' % template_set.lower()
    rs_file = '../templates/swift/%s/resubmit' % template_set.lower()

    # Swift submit file.
    with open(s_file, 'r') as f:
        data = f.read()

    with open('%s/submit' % (data_dir), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))

    # Swift resubmit file.
    with open(rs_file, 'r') as f:
        data = f.read()

    with open('%s/resubmit' % (data_dir), 'w') as f:
        f.write(re.sub('XXX', lambda m, i=iter(r): next(i), data))

    with open('%s/auto_resubmit' % data_dir, 'w') as f:
        f.write('sbatch resubmit')
