# -*- coding: utf-8 -*-
"""
Functions to read MOKE data from HDF5 files

@author: williamrigaut
"""
import h5py
import numpy as np


def get_moke_results(hdf5_file, group_path, result_type=None):
    """
    Reads the results of a MOKE measurement from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    group_path : str or Path
        The path within the HDF5 file to the group containing the MOKE data.
    result_type : str, optional
        The type of result to retrieve. If None, all results are returned. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing the results of the MOKE measurement. If result_type is specified,
        the function returns the value of the corresponding key in the dictionary.
        If the key is not found, the function returns 1.
    """
    results_moke = {}
    units_results_moke = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            node = h5f[group_path]
            for key in node.keys():
                if isinstance(node[key], h5py.Dataset):
                    if node[key].shape == ():
                        results_moke[key] = float(node[key][()])
                    else:
                        results_moke[key] = node[key][()]
                    if "units" in node[key].attrs.keys():
                        units_results_moke[key] = node[key].attrs["units"]

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    if result_type is not None:
        if result_type.lower() in results_moke.keys():
            return (
                results_moke[result_type.lower()],
                units_results_moke[result_type.lower()],
            )

    return results_moke, units_results_moke


def get_moke_loop(hdf5_file, group_path):
    """
    Reads the MOKE loop data from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    group_path : str or Path
        The path within the HDF5 file to the group containing the MOKE loop data.

    Returns
    -------
    dict
        A dictionary containing the MOKE loop data with keys 'applied_field' and 'magnetization'.
    """
    measurement = {}
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            measurement["applied field"] = h5f[group_path]["applied field"][()]
            measurement["magnetization"] = h5f[group_path]["magnetization"][()]
    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return measurement
