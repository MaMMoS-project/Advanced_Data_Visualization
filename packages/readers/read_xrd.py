# -*- coding: utf-8 -*-
"""
Functions to read XRD data from HDF5 files

@author: williamrigaut
"""
import h5py


def _get_attrs(name, obj):
    """
    Used for visit_items() to display all the subgroups and dataset
    Disclaimer: functions starting with '_' are not made to be used by the user unless you know what you're doing
    """
    global attrs

    if isinstance(obj, h5py.Dataset):
        dataset = obj[()]
        attrs[name] = dataset


def get_xrd_results(hdf5_file, group_path, result_type):
    """
    Read the XRD results inside a hdf5 datafile, result_type has to be specified.
    selected_result_type can be 'Phases', 'Global_Parameters' or 'R_coefficients'

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
    attrs = {}
    parent_attrs = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            result_types = h5f[group_path].keys()
            for result in result_types:
                if result_type.lower() in result:
                    result_group = h5f[f"{group_path}/{result}"]
                    for elm in result_group:
                        result_group[elm].visititems(_get_attrs)
                        # Retrieve all the elements of the group and put them in the parent dictionary
                        parent_attrs[elm] = attrs
                        attrs = {}

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return parent_attrs


def get_xrd_pattern(hdf5_file, group_path):

    measurement = {}
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            measurement["counts"] = h5f[group_path]["counts"][()]
            measurement["angle"] = h5f[group_path]["angle"][()]
    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return measurement
