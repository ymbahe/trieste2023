import numpy as np
import hydrangea as hy
import os
import h5py as h5

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
    mstar = hy.hdf5.read_data(sim.fgt_loc, 'Mstar30kpc')[:, isnap]
    pos = hy.hdf5.read_data(sim.gps_loc, 'Centre')[:, isnap, :]
    contflag = hy.hdf5.read_data(sim.fgt_loc, 'ContFlag')[:, isnap]

    cl_rad = np.linalg.norm(pos - cl_pos, axis=1)
    ind = np.nonzero(
        (mstar > 9.0) & (contflag <= 1) & (cl_rad <= 10.0 * r200))[0]
    print(f"Found {len(ind)} galaxies for simulation {sim.run_dir}.")

    galaxy_data = {}
    galaxy_data['ID'] = ind
    galaxy_data['ClusterCentricRadii'] = cl_rad[ind]
    galaxy_data['M200'] = np.zeros(len(ind)) + m200
    galaxy_data['R200'] = np.zeros(len(ind)) + r200

    # [TO DO: add other properties if needed]

    galaxy_data['sim'] = sim
    
    return galaxy_data


def write_galaxy_data(sim_data):
    """Write all the accumulated data to the HDF5 output file."""

    with h5.File(output_file, 'a') as f:
        for key in sim_data.keys():
            f[key] = sim_data[key]


def get_catalog_data(sim_data):
    """Extract galaxy data directly from existing catalogues."""
    pass
    

def determine_environment(sim_data):
    """Determine the environment measurements of galaxïes."""
    pass


def measure_sfr_properties(sim_data):
    """Measure the properties of star-forming gas."""
    pass


def measure_stellar_properties(sim_data):
    """Measure the properties of the stars."""
    pass


def measure_hi_properties(sim_data):
    """Measure properties of HI gas."""
    pass
            

if __name__ == "__main__":
    main()
