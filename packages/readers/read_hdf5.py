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
    """
    Retrieves all unique positions and associated scan numbers from an HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to retrieve positions for, corresponds to a subgroup under 'entry'.

    Returns
    -------
    list of tuples
        A sorted list of unique tuples, each containing the x position, y position, and scan number.
    """

    positions = []

    with h5py.File(hdf5_file, "r") as h5f:
        root_group = f"./entry/{data_type.lower()}"

        # Getting all the positions and the associated scan numbers
        for group in h5f[root_group].keys():
            instrument = h5f[f"{root_group}/{group}"]["instrument"]
            x = instrument["x_pos"][()]
            y = instrument["y_pos"][()]
            scan_number = group.split("_")[1]

            positions.append((x, y, scan_number))

    return sorted(set(positions))


def _get_position_units(hdf5_file, data_type: str):
    """
    Reads the units of the x and y positions from the HDF5 file.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.

    Returns
    -------
    dict
        A dictionary containing the units for the x and y positions.
    """
    position_units = {}

    with h5py.File(hdf5_file, "r") as h5f:
        root_group = f"./entry/{data_type.lower()}"

        # Getting the units for one scan
        for group in h5f[root_group].keys():
            instrument = h5f[f"{root_group}/{group}"]["instrument"]
            position_units["x_pos"] = instrument["x_pos"].attrs["units"]
            position_units["y_pos"] = instrument["y_pos"].attrs["units"]

    return position_units


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
    """
    Construct the xarray Dataset with composition, coercivity, and lattice parameter.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    exclude_wafer_edges : bool, default=True
        If True, exclude the positions at the edges of the wafer (i.e. at x=+/-40 and y=+/-40).

    Returns
    -------
    data : xarray.Dataset
        An xarray Dataset containing the composition, coercivity and lattice parameters of the samples at each position.
    """
    # Looking for EDX positions and scan numbers
    positions = _get_all_positions(hdf5_file, data_type="EDX")
    position_units = _get_position_units(hdf5_file, data_type="EDX")

    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    data = xr.Dataset()

    # Retrieve EDX composition
    for x, y, nb_scan in positions:
        if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
            continue
        edx_group_path = make_group_path(["EDX", nb_scan, "Results"])
        composition, composition_units = get_edx_composition(hdf5_file, edx_group_path)

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

            # Getting the composition units in the xarray
            if "AtomPercent" in composition_units[element].keys():
                data[element_key].attrs["units"] = composition_units[element][
                    "AtomPercent"
                ]

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
        # Setting the values for the coercivity in the xarray with the units
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
        xrd_phases, xrd_units = get_xrd_results(
            hdf5_file, xrd_group_path, result_type="Phases"
        )

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

    # Getting the lattice units in the xarray
    for phase in xrd_phases.keys():
        for i, elm in enumerate(["A", "B", "C"]):
            if elm in phase_keys and elm in xrd_units[phase]:
                data[lattice_labels[i]].attrs["units"] = xrd_units[phase][elm]

    # Setting the units for x_pos and y_pos
    data["x"].attrs["units"] = position_units["x_pos"]
    data["y"].attrs["units"] = position_units["y_pos"]

    return data


def search_measurement_data_from_type(hdf5_file, data_type, nb_scan):
    """
    Search for measurement data of a given type and scan number in a HDF5 file.

    Parameters
    ----------
    hdf5_file : str
        Path to the HDF5 file
    data_type : str
        Type of measurement data to search for. Can be "EDX", "MOKE" or "XRD".
    nb_scan : int
        Number of the scan to search for

    Returns
    -------
    data : xarray.DataArray
        The measurement data
    data_units : dict
        A dictionary with the units of each dimension of the data

    Notes
    -----
    The group path is built using the data_type and nb_scan values. For EDX and XRD, the group path is
    "<data_type>/<nb_scan>/Measurement". For MOKE, the group path is "<data_type>/<nb_scan>/Results".
    """
    group_path = make_group_path([data_type, nb_scan, "Measurement"])

    if data_type.lower() == "edx":
        data, data_units = get_edx_spectrum(hdf5_file, group_path)
    elif data_type.lower() == "moke":
        group_path = make_group_path([data_type, nb_scan, "Results"])
        data, data_units = get_moke_loop(hdf5_file, group_path)
    elif data_type.lower() == "xrd":
        data, data_units = get_xrd_pattern(hdf5_file, group_path)

    return data, data_units


def newDataArray(x_vals, y_vals):
    """
    Create a new DataArray with NaN values for the given x and y coordinates.

    Parameters
    ----------
    x_vals, y_vals : array-like
        The x coordinates and y coordinates

    Returns
    -------
    xr.DataArray
        The new DataArray with NaN values
    """
    return xr.DataArray(np.nan, coords=[y_vals, x_vals], dims=["y", "x"])


def add_measurement_data(dataset, measurement, data_type, x, y, x_vals, y_vals):
    """
    Add measurement data to the given dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to which the measurement data should be added.
    measurement : dict
        A dictionary containing the measurement data.
    data_type : str
        The type of the measurement data (e.g. "edx", "moke", "xrd").
    x : int
        The x position of the measurement.
    y : int
        The y position of the measurement.
    x_vals : list
        A list of all x values in the dataset.
    y_vals : list
        A list of all y values in the dataset.

    Returns
    -------
    None
    """
    if data_type.lower() == "edx":
        if "counts" not in dataset:
            dataset["counts"] = xr.DataArray(
                np.nan,
                coords=[y_vals, x_vals, measurement["energy"]],
                dims=["y", "x", "energy"],
            )
        dataset["counts"].loc[{"y": y, "x": x, "energy": measurement["energy"]}] = (
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
        if "counts" not in dataset:
            dataset["counts"] = xr.DataArray(
                np.nan,
                coords=[y_vals, x_vals, measurement["angle"]],
                dims=["y", "x", "angle"],
            )
        dataset["counts"].loc[{"y": y, "x": x, "angle": measurement["angle"]}] = (
            measurement["counts"]
        )

    return None


def get_current_dataset(data_type, dataset_edx, dataset_moke, dataset_xrd):
    """
    Returns the current dataset for the given data_type from the three given datasets.

    Parameters
    ----------
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD' or 'all'.
    dataset_edx : xarray.DataTree
        The DataTree object containing the EDX data.
    dataset_moke : xarray.DataTree
        The DataTree object containing the MOKE data.
    dataset_xrd : xarray.DataTree
        The DataTree object containing the XRD data.

    Returns
    -------
    xarray.DataTree
        The DataTree object containing the data for the given data_type.
    """
    if data_type.lower() == "edx":
        current_dataset = dataset_edx
    elif data_type.lower() == "moke":
        current_dataset = dataset_moke
    elif data_type.lower() == "xrd":
        current_dataset = dataset_xrd

    return current_dataset


def get_measurement_data(hdf5_file, data_type, exclude_wafer_edges=True):
    """
    Reads the measurement data from an HDF5 file and returns an xarray DataTree object containing all the scans of every experiment.

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD' or 'all'.
    exclude_wafer_edges : bool, optional
        If True, the function will exclude the data measured at the edges of the wafer from the returned DataTree. Defaults to True.

    Returns
    -------
    xarray.DataTree
        A DataTree object containing all the scans of every experiment. The DataTree has a name attribute set to "Measurement Data".
    """
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
            measurement, units = search_measurement_data_from_type(
                hdf5_file, data_type, nb_scan
            )
            current_dataset = get_current_dataset(
                data_type, dataset_edx, dataset_moke, dataset_xrd
            )

            add_measurement_data(
                current_dataset, measurement, data_type, x, y, x_vals, y_vals
            )

        # Add units for x, y positions for all datasets
        position_units = _get_position_units(hdf5_file, data_type=data_type)
        current_dataset["x"].attrs["units"] = position_units["x_pos"]
        current_dataset["y"].attrs["units"] = position_units["y_pos"]

        # Add units for scan axis in all datasets
        # print(units.keys())
        for key in units.keys():
            if data_type.lower() != "moke":
                if key in current_dataset:
                    current_dataset[key].attrs["units"] = units[key]
            else:
                if (
                    key in current_dataset["index_value"]
                    and "units" not in current_dataset["Loops"].attrs
                ):
                    print(units[key])
                    current_dataset["Loops"].attrs["units"] = units

    measurement_tree["EDX"] = dataset_edx
    measurement_tree["MOKE"] = dataset_moke
    measurement_tree["XRD"] = dataset_xrd

    return measurement_tree
