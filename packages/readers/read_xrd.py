# -*- coding: utf-8 -*-
"""
Functions to read XRD data from HDF5 files

@author: williamrigaut
"""
import h5py


def get_xrd_results(hdf5_file, group_path, result_type):
    """
    Read the XRD results inside a hdf5 datafile, result_type has to be specified.
    result_type can be 'Phases', 'Global Parameters' or 'R coefficients'

    Parameters
    ----------
    hdf5_file : STR or pathlib.Path
        Full path to the hdf5 file containing the data to be extracted.
    group_path : STR or pathlib.Path
        Path WITHIN the hdf5 file to the group where the metadata is located,
        e.g. "./EDX/Spectrum_(20, -30)".
    result_type : STR
        values for result_type can be 'Phases', 'Global Parameters' or 'R coefficients'.

    Returns
    -------
    attrs : numpy.array
        List containing xrd results, output will depend on result_type value
    """

    global attrs
    attrs = []

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            keys = h5f[group_path].keys()
            for key in keys:
                if result_type in key:
                    h5f[f"{group_path}/{key}"].visititems(_get_attrs)

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return attrs
