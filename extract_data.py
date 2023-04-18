import numpy as np
import hydrangea as hy
import os
import h5py as h5

from scipy.spatial import cKDTree

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
    pos = hy.hdf5.read_data(cantor_file, 'Subhalo/Position')
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

    mstar_all = hy.hdf5.read_data(cantor_file, 'Subhalo/MassType')[:, 4]
    pos_all = hy.hdf5.read_data(cantor_file, 'Subhalo/Position')

    tree_all = cKDTree(pos_all - sim_data['Temp']['ClusterCoordinates'])
    #tree_targ = cKDTree(sim_data['Coordinates'])

    n_gal = len(sim_data['ID'])
    ngbs = tree_all.query(sim_data['Coordinates'], k=5)

    set_trace()

    for igal in range(n_gal):
        pass


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
            

if __name__ == "__main__":
    main()
