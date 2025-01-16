# -*- coding: utf-8 -*-
"""
Functions to read MOKE data from HDF5 files

@author: williamrigaut
"""
import h5py


def get_moke_results(hdf5_file, group_path, result_type=None):
    """
    Read the MOKE results inside a hdf5 datafile, result_type can be 'Coercivity' or 'Reflectivity'

    Parameters
    ----------
    hdf5_file : STR or pathlib.Path
        Full path to the hdf5 file containing the data to be extracted.
    group_path : STR or pathlib.Path
        Path WITHIN the hdf5 file to the group where the metadata is located,
        e.g. "./EDX/Spectrum_(20, -30)".

    Returns
    -------
    results_moke : numpy.array
        List containing result_type group, or all is result_type was not specified
    """

    results_moke = []
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            keys = h5f[group_path].keys()
            for key in keys:
                dataset = h5f[f"{group_path}/{key}"]
                if result_type is None:
                    results_moke.append([key, dataset[()][0]])
                elif result_type.lower() in key.lower():
                    return [key, dataset[()][0]]
                else:
                    print("Warning, result type was not found.")
                    return 1
    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return results_moke
