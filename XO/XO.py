# Make it as flexible as possible!!!
## you can select float32 and float64
## you can very flexible select what should be stored and would should be retrieved!!
## you can also add your own functions
## analyze data; visualize data; connect with deep learning model, maybe chemical space plot/molecular networking
## connect to semantic lab


import zarr
import numpy as np

from matchms import Spectrum

from matchms.logging_functions import set_matchms_logger_level
set_matchms_logger_level("ERROR")


# TODO: Make a class out of it
# TODO: Make ms1 not mandatory
# Make it jupyter notebook 

def get_ms2_level(spectra):
    """Generator to yield m/z and intensities for MS2 level."""
    for spectrum in spectra:
        mz = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        yield mz, intensities


def get_info(spectra, info_key):
    """Generator to yield user defined info from spectra"""
    for spectrum in spectra:
        info_value = spectrum.get(info_key)
        yield info_value


def generate_indices(spectra):
    """Generate indices to store MS2 data for each spectrum."""
    index_array = np.zeros(len(spectra) + 1, dtype=int)
    spectrum_lengths = np.array([len(spectrum.peaks.mz) for spectrum in spectra])
    index_array[1:] = np.cumsum(spectrum_lengths)
    return index_array


def store(filename, spectra):
    """Store MS1 and MS2 data in Zarr format."""
    # Calculate total sizes (without holding entire dataset in memory)
    total_mz_size = sum(len(spectrum.peaks.mz) for spectrum in spectra)  # Total MS2 data size
    total_spec_size = len(spectra)  # Total number of spectra
    
    # Open Zarr store for writing
    storage = zarr.open(filename, mode="w")
    chunk_size = 1500000  # Define a chunk size for writing data
    
    # Create datasets in Zarr with predefined chunking
    ms2_mz = storage.create_dataset("ms2_mz",
                                    shape=(total_mz_size,),
                                    dtype=np.float64,
                                    chunks=(chunk_size,),
                                    compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=2))
    
    ms2_intensities = storage.create_dataset("ms2_intensities",
                                             shape=(total_mz_size,),
                                             dtype=np.float64,
                                             chunks=(chunk_size,),
                                             compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=2))
    
    ms1_mz = storage.create_dataset("ms1_mz",
                                    shape=(total_spec_size,),
                                    dtype=np.float64,
                                    chunks=(chunk_size,),
                                    compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=2))

    # function for that
    ms2_mz.attrs["author"] = "Jonas Dietrich"
    ms2_mz.attrs["id"] = "x"
    ms2_mz.attrs["instrument_type"] = "Orbitrap"
    ms2_mz.attrs["sample_id"] = "1"

    # Generate and store index array (cumulative MS2 spectrum indices)
    _index_array = generate_indices(spectra)
    index_array = storage.create_dataset("index_array", shape=(total_spec_size + 1,), dtype=np.int32, chunks=(chunk_size,), compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=2))
    
    index_array[:] = _index_array

    ms2_mz_data = []
    ms2_intensities_data = []
    
    for mz, intensities in get_ms2_level(spectra):
        ms2_mz_data.extend(mz)
        ms2_intensities_data.extend(intensities)

    # Store MS2 data
    ms2_mz[:] = ms2_mz_data
    ms2_intensities[:] = ms2_intensities_data

    # Process MS1 data: precursor m/z values

    ms1_mz_data = []
    
    for mz in get_info(spectra, "precursor_mz"):
        ms1_mz_data.append(mz)

    # Store MS1 data
    ms1_mz[:] = ms1_mz_data

    


def read(filename):

    storage = zarr.open(filename, "r")

    ms1_mz = storage["ms1_mz"][:]

    ms2_mz = storage["ms2_mz"][:]
    ms2_intensities = storage["ms2_intensities"][:]

    index_array = storage["index_array"][:]

    spectra = []

    for i in range(len(ms1_mz)):
        start_idx = index_array[i]
        end_idx = index_array[i + 1]

        mz_ms2 = ms2_mz[start_idx:end_idx]
        intensities_ms2 = ms2_intensities[start_idx:end_idx]
        spectrum = Spectrum(mz=mz_ms2,
                            intensities=intensities_ms2,
                            metadata={"precursor_mz": ms1_mz[i]})

        spectra.append(spectrum)

    return spectra
    
        
    

if __name__ == "__main__":
    from time import time
    from matchms.importing import load_from_mgf
    # Load spectra from MGF file
    filename = input("filename.zarr please: ")
    one = time()
    spectra = list(load_from_mgf("small_GNPS-SUSPECTLIST.mgf"))
    two = time()
    print(two - one)
    
    # Store the spectra data in Zarr file
    three = time()
    store(filename, spectra)
    print(three - two)

    spectra = read(filename)
    print(time() - three)
    print(spectra[0])
    
