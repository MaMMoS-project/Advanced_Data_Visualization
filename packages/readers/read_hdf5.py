# -*- coding: utf-8 -*-
"""
This code is a Python module that provides functionality for reading 
HDF5 files for high-throughput experiment.

@author: williamrigaut
"""

import h5py
import xarray as xr
import numpy as np
from packages.readers.read_edx import get_edx_composition
from packages.readers.read_moke import get_moke_results
from packages.readers.read_xrd import get_xrd_results


def _get_attrs(name, obj):
    """
    Used for visit_items() to display all the subgroups and dataset
    Disclaimer: functions starting with '_' are not made to be used by the user unless you know what you're doing
    """
    global attrs

    if isinstance(obj, h5py.Dataset):
        dataset = obj[()]
        attrs.append([name, dataset])


def _get_all_positions(hdf5_file, data_type: str):
    positions = []

    with h5py.File(hdf5_file, "r") as h5f:
        root_group = f"./entry/{data_type.lower()}"

        for group in h5f[root_group].keys():
            instrument = h5f[f"{root_group}/{group}"]["instrument"]
            x = instrument.attrs["x_pos"]
            y = instrument.attrs["y_pos"]
            scan_number = group.split("_")[1]

            positions.append((x, y, scan_number))

    return sorted(set(positions))


def make_group_path(widget_values):
    """
    Write the correct group_path inside the hdf5 datafile depending on the experiment type, the position and data type

    Parameters
    ----------
    widget_values : LIST containing four elements [root_type, x, y, data_type] where
        root_type : STR which can have 'MOKE', 'EDX' or 'XRD' values
        x : INT from -40 to 40 with a step of 5
        y : INT from -40 to 40 with a step of 5
        data_type : STR which can have 'Header', 'Data' or 'Results' values

    Returns
    -------
    group_path : STR
        returns hdf5 group path pointing to the desired location with the input arguments
    """
    root = widget_values[0].lower()
    scan_nb = widget_values[1]
    exp_group = widget_values[2].lower()

    group_path = f"./entry/{root}/scan_{scan_nb}/{exp_group}"

    return group_path


# Construct the xarray Dataset with composition, coercivity, and lattice parameter
def get_full_dataset(hdf5_file):
    positions = _get_all_positions(hdf5_file, data_type="EDX")

    print(positions)

    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    data = xr.Dataset()

    # Retrieve EDX composition
    for x, y, nb_scan in positions:
        edx_group_path = make_group_path(["EDX", nb_scan, "Results"])
        composition = get_edx_composition(hdf5_file, edx_group_path)

        for element in composition:
            print("test1")
            elm_keys = composition[element].keys()
            element_key = f"{element} Composition"
            print("test2")
            if "AtomPercent" not in elm_keys:
                data[element_key] = xr.DataArray(
                    np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                )
                print("test3")
            else:  # NEED ANOTHER CONDITION
                value = composition[element]["AtomPercent"]
                print("test4")
                data[element_key].loc[{"y": y, "x": x}] = value
                print("test5")

    print(data)

    # Retrieve Coercivity (from MOKE results)
    for x, y in positions:
        moke_group_path = dr.make_group_path(["MOKE", str(x), str(y), "Results"])
        coercivity_value = dr.get_moke_results(
            hdf5_file, moke_group_path, result_type="Coercivity"
        )

        if "Coercivity" not in data:
            data["Coercivity"] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
            )
        data["Coercivity"].loc[{"y": y, "x": x}] = coercivity_value[1]

    # Retrieve Lattice Parameter (from XRD results)
    for x, y in positions:
        xrd_group_path = dr.make_group_path(["XRD", str(x), str(y), "Results"])
        lattice_param = dr.get_xrd_results(
            hdf5_file, xrd_group_path, result_type="Phases"
        )

        value_str = lattice_param[0][1][0].decode("utf-8").split("+-")[0]

        # Step 2: Convert to float
        value_float = float(value_str)

        if "Lattice_Parameter" not in data:
            data["Lattice_Parameter"] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
            )
        data["Lattice_Parameter"].loc[{"y": y, "x": x}] = value_float

    return data
