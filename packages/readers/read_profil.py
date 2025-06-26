# -*- coding: utf-8 -*-
"""
Functions to read Profil data from HDF5 files

@author: williamrigaut
"""
import h5py


def get_thickness(hdf5_file, group_path, result_type):
    # Dictionary for DEKTAK results
    profil_attrs = {}
    profil_units = {}

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            results = h5f[group_path].keys()
            for result in results:
                if result == result_type:
                    profil_attrs[result] = h5f[group_path][result][()]
                    profil_units[result] = h5f[group_path][result].attrs["units"]

    except KeyError:
        print("Warning, group path not found in hdf5 file.")
        return 1

    return profil_attrs, profil_units
