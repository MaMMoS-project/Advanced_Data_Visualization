# -*- coding: utf-8 -*-
"""
This code is a Python module that provides functionality for reading 
HDF5 files for high-throughput experiment.

@author: williamrigaut
"""

import h5py
import math
import xarray as xr
import numpy as np
from packages.readers.read_edx import get_edx_composition, get_edx_spectrum
from packages.readers.read_moke import get_moke_results, get_moke_loop
from packages.readers.read_xrd import get_xrd_results, get_xrd_pattern


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
def get_full_dataset(hdf5_file, exclude_wafer_edges=True):
    # Looking for EDX positions and scan numbers
    positions = _get_all_positions(hdf5_file, data_type="EDX")

    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    data = xr.Dataset()

    # Retrieve EDX composition
    for x, y, nb_scan in positions:
        if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
            continue
        edx_group_path = make_group_path(["EDX", nb_scan, "Results"])
        composition = get_edx_composition(hdf5_file, edx_group_path)

        for element in composition:
            elm_keys = composition[element].keys()
            element_key = f"{element} Composition"
            if "AtomPercent" not in elm_keys:
                value = np.nan
            else:
                value = composition[element]["AtomPercent"]

            if element_key not in data and not math.isnan(value):
                data[element_key] = xr.DataArray(
                    np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                )
            if element_key in data:
                data[element_key].loc[{"y": y, "x": x}] = value

    # Looking for MOKE positions and scan numbers
    positions = _get_all_positions(hdf5_file, data_type="MOKE")

    # Retrieve Coercivity (from MOKE results)
    for x, y, nb_scan in positions:
        if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
            continue
        moke_group_path = make_group_path(["MOKE", nb_scan, "Results"])
        coercivity_value, moke_units = get_moke_results(
            hdf5_file, moke_group_path, result_type="Coercivity"
        )

        if "Coercivity" not in data:
            data["Coercivity"] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
            )
        # print(type(coercivity_value))
        if isinstance(coercivity_value, float):
            data["Coercivity"].loc[{"y": y, "x": x}] = coercivity_value
        if isinstance(moke_units, str):
            data["Coercivity"].attrs["units"] = moke_units

    # Looking for XRD positions and scan numbers
    positions = _get_all_positions(hdf5_file, data_type="XRD")

    # Retrieve Lattice Parameter (from XRD results)
    for x, y, nb_scan in positions:
        if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
            continue
        xrd_group_path = make_group_path(["XRD", nb_scan, "Results"])
        # print(xrd_group_path)
        xrd_phases = get_xrd_results(hdf5_file, xrd_group_path, result_type="Phases")

        # Looking for the lattice parameters among all the phases attributs
        for phase in xrd_phases.keys():
            lattice_a = np.nan
            lattice_b = np.nan
            lattice_c = np.nan
            lattice_a_label = f"{phase} Lattice Parameter A"
            lattice_b_label = f"{phase} Lattice Parameter B"
            lattice_c_label = f"{phase} Lattice Parameter C"
            phase_keys = xrd_phases[phase].keys()

            if "A" in phase_keys:
                a_str = str(xrd_phases[phase]["A"])
                if not "UNDEF'" in a_str:
                    lattice_a = a_str.split("+-")[0].replace("b'", "")
                    lattice_a = float(lattice_a)
            if "B" in phase_keys:
                b_str = str(xrd_phases[phase]["B"])
                if not "UNDEF'" in b_str:
                    lattice_b = b_str.split("+-")[0].replace("b'", "")
                    lattice_b = float(lattice_b)
            if "C" in phase_keys:
                c_str = str(xrd_phases[phase]["C"])
                if not "UNDEF'" in c_str:
                    lattice_c = c_str.split("+-")[0].replace("b'", "")
                    lattice_c = float(lattice_c)

            # Adding all the lattice parameters to the dataset
            lattice_labels = [lattice_a_label, lattice_b_label, lattice_c_label]
            lattice_values = [lattice_a, lattice_b, lattice_c]

            for i in range(len(lattice_labels)):
                # Test if all lattice values are not np.nan (if there is no B values we do not create the corresponding array)
                if lattice_labels[i] not in data and not math.isnan(lattice_values[i]):
                    data[lattice_labels[i]] = xr.DataArray(
                        np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                    )
                if lattice_labels[i] in data:
                    data[lattice_labels[i]].loc[{"y": y, "x": x}] = lattice_values[i]

    return data


def search_measurement_data_from_type(hdf5_file, data_type, nb_scan):
    group_path = make_group_path([data_type, nb_scan, "Measurement"])

    if data_type.lower() == "edx":
        data = get_edx_spectrum(hdf5_file, group_path)
    elif data_type.lower() == "moke":
        group_path = make_group_path([data_type, nb_scan, "Results"])
        data = get_moke_loop(hdf5_file, group_path)
    elif data_type.lower() == "xrd":
        data = get_xrd_pattern(hdf5_file, group_path)

    return data


def newDataArray(x_vals, y_vals):
    return xr.DataArray(np.nan, coords=[y_vals, x_vals], dims=["y", "x"])


def add_measurement_data(dataset, measurement, data_type, x, y, x_vals, y_vals):
    if data_type.lower() == "edx":
        if "Spectrum" not in dataset:
            dataset["Spectrum"] = xr.DataArray(
                np.nan,
                coords=[y_vals, x_vals, measurement["energy"]],
                dims=["y", "x", "energy"],
            )
        dataset["Spectrum"].loc[{"y": y, "x": x, "energy": measurement["energy"]}] = (
            measurement["counts"]
        )

    if data_type.lower() == "moke":
        n_indexes = range(len(measurement["applied field"]))

        if "Loops" not in dataset:
            dataset["Loops"] = xr.DataArray(
                np.nan,
                coords=[y_vals, x_vals, ["magnetization", "applied field"], n_indexes],
                dims=["y", "x", "index_value", "n_indexes"],
            )

        dataset["Loops"].loc[
            {"y": y, "x": x, "index_value": "magnetization", "n_indexes": n_indexes}
        ] = measurement["magnetization"]
        dataset["Loops"].loc[
            {"y": y, "x": x, "index_value": "applied field", "n_indexes": n_indexes}
        ] = measurement["applied field"]

    if data_type.lower() == "xrd":
        if "Counts" not in dataset:
            dataset["Counts"] = xr.DataArray(
                np.nan,
                coords=[y_vals, x_vals, measurement["angle"]],
                dims=["y", "x", "angle"],
            )
        dataset["Counts"].loc[{"y": y, "x": x, "angle": measurement["angle"]}] = (
            measurement["counts"]
        )

    return None


def get_current_dataset(data_type, dataset_edx, dataset_moke, dataset_xrd):
    if data_type.lower() == "edx":
        current_dataset = dataset_edx
    elif data_type.lower() == "moke":
        current_dataset = dataset_moke
    elif data_type.lower() == "xrd":
        current_dataset = dataset_xrd

    return current_dataset


def get_measurement_data(hdf5_file, data_type, exclude_wafer_edges=True):
    # Check if data_type is valid
    if data_type.lower() == "all":
        datatypes = ["EDX", "MOKE", "XRD"]

    elif not data_type.lower() in ["edx", "moke", "xrd"]:
        print("data_type must be one of 'EDX', 'MOKE', 'XRD' or 'all'.")
        return 1
    else:
        datatypes = [data_type]

    measurement_tree = xr.DataTree(name="Measurement Data")
    dataset_edx = xr.Dataset()
    dataset_moke = xr.Dataset()
    dataset_xrd = xr.Dataset()

    for data_type in datatypes:
        positions = _get_all_positions(hdf5_file, data_type=data_type)
        x_vals = sorted(set([pos[0] for pos in positions]))
        y_vals = sorted(set([pos[1] for pos in positions]))

        for x, y, nb_scan in positions:
            if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
                continue
            measurement = search_measurement_data_from_type(
                hdf5_file, data_type, nb_scan
            )
            current_dataset = get_current_dataset(
                data_type, dataset_edx, dataset_moke, dataset_xrd
            )

            add_measurement_data(
                current_dataset, measurement, data_type, x, y, x_vals, y_vals
            )

    measurement_tree["EDX"] = dataset_edx
    measurement_tree["MOKE"] = dataset_moke
    measurement_tree["XRD"] = dataset_xrd

    return measurement_tree
