import numpy as np
import hydrangea as hy
import os
import h5py as h5

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

from pdb import set_trace

# Name of the output file -- one for everything
output_file = 'Trieste_Hydrangea.hdf5'

# Snapshot number to analyse -- hardcode this for now.
isnap = 29

def main():
    """Main function."""

    prepare_output()

    full_data = None
    
    for isim in range(1):

        # Skip bad/non-existing simulations
        if isim in [10, 17, 19, 20, 23, 26, 27]:
            continue

        sim_data = process_sim(isim)
        del(sim_data['sim'])
        del(sim_data['Temp'])
        
        sim_data['Sim'] = np.zeros(len(sim_data['ID']), dtype=int) + isim
        
        if full_data is None:
            full_data = {}
            for key in sim_data:
                full_data[key] = sim_data[key]
        else:
            for key in sim_data:
                full_data[key] = np.concatenate(
                    (full_data[key], sim_data[key]))
        
        
    # Write data to the output file
    write_galaxy_data(full_data)

        

def prepare_output():
    """Set up the output file."""
    if os.path.isfile(output_file):
        os.rename(output_file, output_file + '.bak')


def process_sim(isim):
    """High-level function to process one simulation."""
    sim = hy.Simulation(isim)

    # Find the galaxies and gather some first (basic) data about them.
    # 'sim_data' is a dict.
    sim_data = find_galaxies(sim)

    # Data that are already in the Cantor catalogue
    get_catalog_data(sim_data)

    # Environment classification
    determine_environment(sim_data)

    # SFR measurements
    measure_sfr_properties(sim_data)

    # Stellar measurements
    measure_stellar_properties(sim_data)

    # HI measurements
    measure_hi_properties(sim_data) 

    return sim_data
    

def find_galaxies(sim):
    """Find relevant galaxies for one simulation.

    We want things within 10 r200 from the central cluster (FOF-0).
    """
    fof = hy.SplitFile(sim.get_subfind_file(isnap), 'FOF', read_index=0)
    cl_pos = fof.GroupCentreOfPotential
    r200 = fof.Group_R_Crit200
    m200 = fof.Group_M_Crit200

    # Now find galaxies...
    cantor_file = sim.high_level_dir + f'/Cantor/Cantor_{isnap:03d}.hdf5'

    mstar = hy.hdf5.read_data(cantor_file, 'Subhalo/MassType')[:, 4] * 1e10
    pos = hy.hdf5.read_data(cantor_file, 'Subhalo/CentreOfPotential')
    gid = hy.hdf5.read_data(cantor_file, 'Subhalo/Galaxy')
    gal_contflag = hy.hdf5.read_data(sim.fgt_loc, 'ContFlag')[:, isnap]

    cl_rad = np.linalg.norm(pos - cl_pos, axis=1)
    ind = np.nonzero(
        (mstar > 1e9) & (gal_contflag[gid] <= 1) & (cl_rad <= 10.0*r200))[0]
    n_gal = len(ind)

    print(f"Found {n_gal} galaxies for simulation {sim.run_dir}.")

    galaxy_data = {}
    galaxy_data['ID'] = gid[ind]
    galaxy_data['CantorIndex'] = ind

    galaxy_data['ClusterCentricRadii'] = cl_rad[ind]
    galaxy_data['M200'] = np.zeros(n_gal) + m200
    galaxy_data['R200'] = np.zeros(n_gal) + r200
    galaxy_data['Coordinates'] = pos[ind, :] - cl_pos
    galaxy_data['Mstar'] = mstar[ind]

    # [TO DO: add other properties if needed]

    galaxy_data['sim'] = sim
    galaxy_data['Temp'] = {'ClusterCoordinates': cl_pos}

    return galaxy_data


def write_galaxy_data(sim_data):
    """Write all the accumulated data to the HDF5 output file."""

    with h5.File(output_file, 'a') as f:
        for key in sim_data.keys():
            f[key] = sim_data[key]


def get_catalog_data(sim_data):
    """Extract galaxy data directly from existing catalogues."""
    sim = sim_data['sim']
    gid = sim_data['ID']

    # Subfind data (load from FullGalaxyTables)
    sim_data['Mstar_Subfind_30kpc'] = 10.0**(
        hy.hdf5.read_data(sim.fgt_loc, 'Mstar30kpc', read_index=gid)[:, isnap])
    sim_data['R50star_Subfind'] = hy.hdf5.read_data(
        sim.fgt_loc, 'StellarHalfMassRad', read_index=gid)[:, isnap]

    # Cantor data
    cantor_file = sim.high_level_dir + f'/Cantor/Cantor_{isnap:03d}.hdf5'

    csh = sim_data['CantorIndex']
    iextra = hy.hdf5.read_data(
        cantor_file, 'Subhalo/Extra/ExtraIDs', read_index=csh)
    ind_good = np.nonzero(iextra >= 0)[0]
    sim_data['CantorFlag'] = np.zeros_like(sim_data['ID'])
    sim_data['CantorFlag'][ind_good] = 1

    sim_data['R50star_30kpc'] = hy.hdf5.read_data(
        cantor_file, 'Subhalo/Extra/Stars/QuantileRadii',
        read_index=iextra
    )[:, 0, 1]
    sim_data['Mstar_30kpc'] = hy.hdf5.read_data(
        cantor_file, 'Subhalo/Extra/Stars/ApertureMasses',
        read_index=iextra
    )[:, 2]
    sim_data['Cantor_iextra'] = iextra


def determine_environment(sim_data):
    """Determine the environment measurements of galaxies.
    
    Options: M_DM (excl particles in galaxy itself) within 500 (1000?) kpc
    DM density profile over whole cluster
    Fifth-nearest galaxy with M_star > 10^9 M_Sun
    """

    sim = sim_data['sim']
    cantor_file = sim.high_level_dir + f'/Cantor/Cantor_{isnap:03d}.hdf5'

    mstar_all = hy.hdf5.read_data(cantor_file, 'Subhalo/MassType')[:, 4] * 1e10
    pos_all = hy.hdf5.read_data(cantor_file, 'Subhalo/CentreOfPotential')

    ind = np.nonzero(mstar_all > 1e9)[0]

    tree_all = cKDTree(pos_all[ind, :] - sim_data['Temp']['ClusterCoordinates'])
    #tree_targ = cKDTree(sim_data['Coordinates'])

    n_gal = len(sim_data['ID'])
    ngbs, ngb_inds = tree_all.query(sim_data['Coordinates'], k=6)
    d5 = ngbs[:, -1]

    sim_data['Distance_5thNearestNgb'] = d5


    # ----------------
    # Local DM density
    # ----------------

    rho_dm = np.zeros(n_gal)
    snap_file = sim.get_snapshot_file(isnap)

    cl_pos = sim_data['Temp']['ClusterCoordinates']
    cantor_inds = sim_data['CantorIndex']

    rmax_dm = hy.hdf5.read_data(
        cantor_file, 'Subhalo/MaxRadiusType')[cantor_inds, 1]
    mdm = hy.hdf5.read_data(
        cantor_file, 'Subhalo/MassType')[cantor_inds, 1] * 1e10

    for iigal in range(n_gal):

        pos = sim_data['Coordinates'][iigal, :] + cl_pos
        dm = hy.ReadRegion(snap_file, 1, pos, 1.0, shape='sphere', exact=True)
        dm_tot = dm.num_particles_exact * dm.m_dm

        # Need to calculate the mass of DM in the aperture that belongs
        # to the galaxy itself. This is easy if the galaxy is completely
        # enclosed, otherwise it's a bit of a pain
        if rmax_dm[iigal] <= 1.0:
            dm_gal = mdm[iigal]
        else:
            # Get the DM particle IDs that belong to the galaxy (including
            # any that are outside the aperture), and then match them to the
            # IDs of all particles within the aperture. The intersection is
            # the number of DM particles of the galaxy within the aperture.
            dids, galid, galshi = get_cantor_pids(sim, 1, ish=cantor_inds[iigal],
                use_central=False)
            ind_in_sphere, in_sphere = hy.crossref.find_id_indices(
                    dids, dm.ParticleIDs)
            dm_gal = len(in_sphere) * dm.m_dm

        rho_dm[iigal] = (dm_tot - dm_gal) / (4/3 * np.pi)

    sim_data['DM_Density'] = rho_dm

    # DM density profile over whole cluster
    lim_rad = 10.0 * sim_data['R200'][0]
    dm_cl = hy.ReadRegion(snap_file, 1, cl_pos, lim_rad,
                          shape='sphere', exact=True)
    dm_radii = np.linalg.norm(dm_cl.Coordinates - cl_pos, 1)

    m_dm_bins, edges = np.histogram(
        dm_radii, bins=np.arange(0, lim_rad+0.01, 0.1))
    m_dm_bins = m_dm_bins * dm.m_dm
    mid = (edges[1:] + edges[:-1]) / 2
    vol = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_dm_cl = m_dm_bins / vol

    csi = CubicSpline(mid, rho_dm_cl)

    sim_data['DM_ClusterDensity'] = csi(sim_data['ClusterCentricRadii'])


def measure_sfr_properties(sim_data):
    """Measure the properties of star-forming gas."""
    pass


def measure_stellar_properties(sim_data):
    """Measure the properties of the stars.

    2D stellar half-mass radius, anything else...?
    """
    


def measure_hi_properties(sim_data):
    """Measure properties of HI gas."""
    pass


def get_cantor_pids(
    sim, ptype, igal=None, ish=None, use_central=True, isnap=29):
    """Extract the particle IDs for all stars in one cluster.

    Parameters
    ----------
    sim : Simulation object
        The simulation to process.
    ptype : int
        The particle type for which to extract IDs.
    igal : int, optional
        The Galaxy ID of the cluster central. If None (default), `ish` must
        be specified instead.
    ish : int, optional
        The Cantor subhalo index of the cluster central. If None (default),
        `igal` must be specified.
    use_central : bool, optional
        If True (default), translate the specified galaxy into the (Cantor)
        central.

    Returns
    -------
    ids : ndarray(int)
        The particle IDs of the particles in the selected galaxy.
    igal : int
        The galaxy ID of the galaxy actually analysed.
    ish : int
        The Cantor index of the galaxy actually analysed.
    
    """
    if igal is None:
        if ish is None:
            raise ValueError("Must specify `ish` or `igal`!")
        igal = hy.hdf5.read_data(
            sim.high_level_dir + f'/Cantor/Cantor_{isnap:03d}.hdf5',
            'Subhalo/Galaxy', read_index=ish
        )

    if use_central:
        # Find the galaxy ID of the (corrected) central galaxy of the cluster
        igal = hy.hdf5.read_data(
            sim.high_level_dir + '/Cantor/GalaxyTables.hdf5', 'CentralGalaxy',
            read_index=igal
        )[isnap]
        ish = None   # So we look it up fresh

    if ish is None:
        ish = hy.hdf5.read_data(
            sim.high_level_dir + '/Cantor/GalaxyTables.hdf5', 'SubhaloIndex',
            read_index=igal
        )[isnap]
    
    # Extract the list of all particles in the target galaxy
    cantor_base = sim.high_level_dir + f'/Cantor/Cantor_{isnap:03d}'
    cantor_file = cantor_base + '.hdf5'
    cantor_id_file = cantor_base + '_IDs.hdf5'

    cl_offsets = hy.hdf5.read_data(
        cantor_file, 'Subhalo/OffsetType', read_index=ish)
    id_ind_start = cl_offsets[ptype]
    id_ind_end = cl_offsets[ptype + 1]

    cl_ids = hy.hdf5.read_data(
        cantor_id_file, 'IDs',
        read_index=np.arange(id_ind_start, id_ind_end, dtype=int)
    )

    return cl_ids, igal, ish

if __name__ == "__main__":
    main()
