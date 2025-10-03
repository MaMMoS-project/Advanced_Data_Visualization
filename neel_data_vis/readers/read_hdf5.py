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
from .read_edx import get_edx_composition, get_edx_spectrum
from .read_moke import get_moke_results, get_moke_loop
from .read_xrd import get_xrd_results, get_xrd_pattern, get_xrd_image
from .read_profil import get_thickness
from tqdm import tqdm


def make_group_path(
    hdf5_file, data_type, measurement_type=None, x_pos=None, y_pos=None
):
    """
    Builds the path to the data group in the HDF5 file using the data type and optionally the measurement type, x and y positions.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.
    measurement_type : str, optional
        The type of measurement to read. If not given, the function will only return the group path for the data type.
    x_pos : float, optional
        The x position of the measurement. If not given, the function will only return the group path for the data type.
    y_pos : float, optional
        The y position of the measurement. If not given, the function will only return the group path for the data type.

    Returns
    -------
    str
        The path to the group in the HDF5 file containing the data.
    """
    with h5py.File(hdf5_file, "r") as h5f:
        # Check which group corresponds to the data type
        start_group = []
        for group in h5f["./"]:
            if "HT_type" not in h5f[f"./{group}"].attrs.keys():
                continue

            if data_type is None:
                start_group.append(f"./{group}")
                break

            if h5f[f"./{group}"].attrs["HT_type"] == data_type.lower():
                start_group.append(f"./{group}")

        if len(start_group) == 0:
            print(f"Data type {data_type} not found in HDF5 file.")
            return 1

        if measurement_type is None or x_pos is None or y_pos is None:
            return start_group

        # Getting the corresponding measurement path
        x_pos = str(round(float(x_pos), 1))
        y_pos = str(round(float(y_pos), 1))
        group = f"({x_pos},{y_pos})"

        group_path = []
        for elm_start_group in start_group:
            group_path.append(f"{elm_start_group}/{group}/{measurement_type.lower()}")

    return group_path


def get_all_positions(hdf5_file, data_type: str):
    """
    Reads all positions from a HDF5 file for a given data type.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.

    Returns
    -------
    list
        A list of tuples (x, y) containing all positions present in the HDF5 file for the given data type.
    """
    positions = []
    data_group = make_group_path(hdf5_file, data_type=data_type)[0]

    with h5py.File(hdf5_file, "r") as h5f:
        for group in h5f[data_group]:
            # Skipping scan groupes in MOKE data and alignement scans in ESRF data
            if group in ["scan_parameters", "alignment_scans"]:
                continue

            instrument = h5f[f"{data_group}/{group}/instrument"]
            x = round(float(instrument["x_pos"][()]), 1)
            y = round(float(instrument["y_pos"][()]), 1)
            positions.append((x, y))

    return sorted(set(positions))


def get_position_units(hdf5_file, data_type: str):
    """
    Reads the units of the position coordinates from a HDF5 file for a given data type.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE' or 'XRD'.

    Returns
    -------
    dict
        A dictionary with the units of the x and y coordinates of the positions.
    """
    position_units = {}

    root_group = make_group_path(hdf5_file, data_type=data_type)[0]

    with h5py.File(hdf5_file, "r") as h5f:
        instrument = h5f[f"{root_group}"]["(0.0,0.0)"]["instrument"]
        position_units["x_pos"] = instrument["x_pos"].attrs["units"]
        position_units["y_pos"] = instrument["y_pos"].attrs["units"]

    return position_units


def get_group_path_label(main_group_path, sub_group_path):
    if len(main_group_path) > 1:
        group_path_label = sub_group_path.split("/")[-3]
    else:
        group_path_label = None

    return group_path_label


def set_edx_composition(
    data, composition, composition_units, x, y, x_vals, y_vals, group_path_label=None
):
    for element in composition:
        elm_keys = composition[element].keys()

        if group_path_label is None:
            element_key = f"{element} Composition"
        else:
            element_key = f"{group_path_label} {element} Composition"

        # Getting the composition values in the xarray
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
            data[element_key].attrs["units"] = composition_units[element]["AtomPercent"]

    return data


def set_moke_values(
    data, moke_value, moke_units, x, y, x_vals, y_vals, group_path_label=None
):
    for value in moke_value:
        if group_path_label is None:
            label_value = value
        else:
            label_value = f"{group_path_label} {value}"

        if label_value not in data:
            data[label_value] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
            )

        data[label_value].loc[{"y": y, "x": x}] = moke_value[value]
        data[label_value].attrs["units"] = moke_units[value]

    return data


def set_xrd_phases(
    data, xrd_phases, xrd_units, x, y, x_vals, y_vals, group_path_label=None
):
    # Looking for the lattice parameters among all the phases attributs
    for phase in xrd_phases.keys():
        phase_fraction = np.nan
        lattice_a = np.nan
        lattice_b = np.nan
        lattice_c = np.nan

        if group_path_label is None:
            phase_fraction_label = f"{phase} Phase Fraction"
            lattice_a_label = f"{phase} Lattice Parameter A"
            lattice_b_label = f"{phase} Lattice Parameter B"
            lattice_c_label = f"{phase} Lattice Parameter C"
        else:
            phase_fraction_label = f"{group_path_label} {phase} Phase Fraction"
            lattice_a_label = f"{group_path_label} {phase} Lattice Parameter A"
            lattice_b_label = f"{group_path_label} {phase} Lattice Parameter B"
            lattice_c_label = f"{group_path_label} {phase} Lattice Parameter C"

        phase_keys = xrd_phases[phase].keys()

        if "phase_fraction" in phase_keys:
            phase_fraction = str(xrd_phases[phase]["phase_fraction"])
            if not "UNDEF'" in phase_fraction:
                phase_fraction = (
                    phase_fraction.split("+-")[0].replace("b", "").replace("'", "")
                )
                phase_fraction = float(phase_fraction)
        if "A" in phase_keys:
            a_str = str(xrd_phases[phase]["A"])
            if not "UNDEF'" in a_str:
                lattice_a = a_str.split("+-")[0].replace("b", "").replace("'", "")
                lattice_a = float(lattice_a)
        if "B" in phase_keys:
            b_str = str(xrd_phases[phase]["B"])
            if not "UNDEF'" in b_str:
                lattice_b = b_str.split("+-")[0].replace("b", "").replace("'", "")
                lattice_b = float(lattice_b)
        if "C" in phase_keys:
            c_str = str(xrd_phases[phase]["C"])
            if not "UNDEF'" in c_str:
                lattice_c = c_str.split("+-")[0].replace("b", "").replace("'", "")
                lattice_c = float(lattice_c)

        # Adding all the lattice parameters to the dataset
        lattice_labels = [
            phase_fraction_label,
            lattice_a_label,
            lattice_b_label,
            lattice_c_label,
        ]
        lattice_values = [phase_fraction, lattice_a, lattice_b, lattice_c]

        for i, label in enumerate(lattice_labels):
            # Test if all lattice values are not np.nan (if there is no B values we do not create the corresponding array)
            if label not in data and not math.isnan(lattice_values[i]):
                data[label] = xr.DataArray(
                    np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                )
                # Adding the units of the new xrd dataset
                if "Phase Fraction" in label:
                    data[label].attrs["units"] = "Wt.%"
                elif "Lattice Parameter" in label:
                    lattice_param = label.split()[-1]
                    data[label].attrs["units"] = xrd_units[phase][lattice_param]

            if label in data:
                data[label].loc[{"y": y, "x": x}] = lattice_values[i]

    return data


def get_full_dataset(hdf5_file, exclude_wafer_edges=True):
    """
    Reads the measurement data from an HDF5 file and returns an xarray DataArray object containing all the scans of every experiment.

    Parameters
    ----------
    hdf5_file : str or Path
        The path to the HDF5 file to read the data from.
    exclude_wafer_edges : bool, optional
        If True, the function will exclude the data measured at the edges of the wafer from the returned DataArray. Defaults to True.

    Returns
    -------
    xarray.DataArray
        A DataArray object containing all the scans of every experiment. The DataArray has a name attribute set to "Measurement Data".
    """

    # Looking for EDX positions and scan numbers
    positions = get_all_positions(hdf5_file, data_type=None)
    position_units = get_position_units(hdf5_file, data_type=None)

    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    data = xr.Dataset()

    # Retrieve EDX composition
    try:
        positions = get_all_positions(hdf5_file, data_type="EDX")
        for x, y in positions:
            if (
                (np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges)
                or np.abs(x) > 40
                or np.abs(y) > 40
            ):
                continue

            edx_group_path = make_group_path(
                hdf5_file, x_pos=x, y_pos=y, data_type="EDX", measurement_type="Results"
            )

            # Checking if there are multiple EDX scans
            for group_path in edx_group_path:
                group_path_label = get_group_path_label(edx_group_path, group_path)

                try:
                    composition, composition_units = get_edx_composition(
                        hdf5_file, group_path
                    )
                    # Setting the composition values in the xarray
                    data = set_edx_composition(
                        data,
                        composition,
                        composition_units,
                        x,
                        y,
                        x_vals,
                        y_vals,
                        group_path_label=group_path_label,
                    )

                except KeyError:
                    # Skipping the scan without EDX data
                    pass

    except (KeyError, TypeError):
        print("Warning: No EDX results found in the file")
        pass

    # Retrieve Coercivity (from MOKE results)
    try:
        # Looking for MOKE positions and scan numbers
        positions = get_all_positions(hdf5_file, data_type="MOKE")
        for x, y in positions:
            if (
                (np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges)
                or np.abs(x) > 40
                or np.abs(y) > 40
            ):
                continue

            moke_group_path = make_group_path(
                hdf5_file,
                x_pos=x,
                y_pos=y,
                data_type="MOKE",
                measurement_type="Results",
            )

            # Checking if there are multiple MOKE scans
            for group_path in moke_group_path:
                group_path_label = get_group_path_label(moke_group_path, group_path)

                try:
                    moke_value, moke_units = get_moke_results(
                        hdf5_file, group_path, result_type=None
                    )
                    # Setting the magnetic values in the xarray
                    data = set_moke_values(
                        data,
                        moke_value,
                        moke_units,
                        x,
                        y,
                        x_vals,
                        y_vals,
                        group_path_label=group_path_label,
                    )
                except KeyError:
                    # Skipping the scans with no results
                    pass

    except (KeyError, TypeError):
        print("Warning: No MOKE results found in the file")
        pass

    try:
        # Looking for XRD positions and scan numbers
        positions = get_all_positions(hdf5_file, data_type="XRD")

        # Retrieve Lattice Parameter (from XRD results)
        for x, y in positions:
            if (
                (np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges)
                or np.abs(x) > 40
                or np.abs(y) > 40
            ):
                continue

            xrd_group_path = make_group_path(
                hdf5_file, x_pos=x, y_pos=y, data_type="XRD", measurement_type="Results"
            )

            # Checking if there are multiple XRD scans
            for group_path in xrd_group_path:
                group_path_label = get_group_path_label(xrd_group_path, group_path)

                try:
                    xrd_phases, xrd_units = get_xrd_results(
                        hdf5_file, group_path, result_type="Phases"
                    )
                    data = set_xrd_phases(
                        data,
                        xrd_phases,
                        xrd_units,
                        x,
                        y,
                        x_vals,
                        y_vals,
                        group_path_label=group_path_label,
                    )
                except KeyError:
                    # Skipping the scans with no results
                    pass

    except (KeyError, TypeError):
        print("Warning: No XRD results found in the file")
        pass

    try:
        # Looking for PROFIL positions and scan numbers
        positions = get_all_positions(hdf5_file, data_type="PROFIL")
        for x, y in positions:
            if (
                (np.abs(x) + np.abs(y) >= 60 and exclude_wafer_edges)
                or np.abs(x) > 40
                or np.abs(y) > 40
            ):
                continue
            profil_group_path = make_group_path(
                hdf5_file,
                x_pos=x,
                y_pos=y,
                data_type="PROFIL",
                measurement_type="Results",
            )
            profil_results, profil_units = get_thickness(
                hdf5_file, group_path=profil_group_path, result_type="measured_height"
            )

            for value in profil_results.keys():
                if value not in data:
                    data[value] = xr.DataArray(
                        np.nan, coords=[y_vals, x_vals], dims=["y", "x"]
                    )
                data[value].loc[{"y": y, "x": x}] = profil_results[value]
                data[value].attrs["units"] = profil_units[value]
    except (KeyError, TypeError):
        print("Warning: No PROFIL results found in the file")
        pass

    # Setting the units for x_pos and y_pos
    data["x"].attrs["units"] = position_units["x_pos"]
    data["y"].attrs["units"] = position_units["y_pos"]

    return data


def search_measurement_data_from_type(hdf5_file, data_type, x_pos, y_pos):
    """
    Retrieves measurement data from an HDF5 file for a specified data type and position.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read, either 'EDX', 'MOKE', or 'XRD'.
    x_pos : float
        The x position of the measurement.
    y_pos : float
        The y position of the measurement.

    Returns
    -------
    tuple
        A tuple containing the measurement data and its units.
    """
    data = {}
    data_units = {}

    if data_type.lower() == "edx":
        edx_group_path = make_group_path(
            hdf5_file,
            data_type="EDX",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        for group in edx_group_path:
            group_path_label = get_group_path_label(edx_group_path + [""], group)
            data[group_path_label], data_units[group_path_label] = get_edx_spectrum(
                hdf5_file, group
            )

    elif data_type.lower() == "moke":
        moke_group_path = make_group_path(
            hdf5_file,
            data_type="MOKE",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        for group in moke_group_path:
            group_path_label = get_group_path_label(moke_group_path + [""], group)
            data[group_path_label], data_units[group_path_label] = get_moke_loop(
                hdf5_file, group
            )

    elif data_type.lower() == "xrd":
        xrd_group_path = make_group_path(
            hdf5_file,
            data_type="XRD",
            measurement_type="Measurement",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        for group in xrd_group_path:
            group_path_label = get_group_path_label(xrd_group_path + [""], group)
            data[group_path_label], data_units[group_path_label] = get_xrd_pattern(
                hdf5_file, group
            )

    return data, data_units


def add_measurement_data(
    dataset, measurement, data_type, x, y, x_vals, y_vals, meas_key
):
    """
    Adds a measurement data point to the given dataset. The dataset should have the structure
    of an xarray DataArray.

    Parameters
    ----------
    dataset : xarray.DataArray
        The dataset to add the measurement data point to.
    measurement : dict
        A dictionary containing the measurement data.
    data_type : str
        The type of measurement, either 'EDX', 'MOKE', or 'XRD'.
    x : float
        The x position of the measurement.
    y : float
        The y position of the measurement.
    x_vals : list
        A list of all the x values in the dataset.
    y_vals : list
        A list of all the y values in the dataset.

    Returns
    -------
    None
    """

    for key in measurement.keys():
        data = measurement[key]

        if key not in dataset:
            dataset[key] = xr.DataArray(
                np.nan, coords=[y_vals, x_vals, data], dims=["y", "x", key]
            )
        else:
            dataset[key].loc[{"y": y, "x": x}] = data

    return None


def get_current_dataset(data_type, dataset_edx, dataset_moke, dataset_xrd):
    """
    Returns the current dataset based on the given data_type.

    Parameters
    ----------
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD'.
    dataset_edx : xarray.Dataset
        The Dataset containing the EDX data.
    dataset_moke : xarray.Dataset
        The Dataset containing the MOKE data.
    dataset_xrd : xarray.Dataset
        The Dataset containing the XRD data.

    Returns
    -------
    xarray.Dataset
        The current dataset based on the given data_type.
    """

    if data_type.lower() == "edx":
        current_dataset = dataset_edx
    elif data_type.lower() == "moke":
        current_dataset = dataset_moke
    elif data_type.lower() == "xrd":
        current_dataset = dataset_xrd

    return current_dataset


def get_measurement_data(hdf5_file, datatype, meas_key=None, exclude_wafer_edges=True):
    """
    Reads measurement data from the given HDF5 file and returns an xarray DataTree object containing the measurement data.

    Parameters
    ----------
    hdf5_file : str or pathlib.Path
        The path to the HDF5 file to read the data from.
    data_type : str
        The type of data to read. Must be one of 'EDX', 'MOKE', 'XRD'
    exclude_wafer_edges : bool, optional
        If True, the function will exclude the data measured at the edges of the wafer from the returned DataTree. Defaults to True.

    Returns
    -------
    xarray.DataTree
        A DataTree object containing the measurement data.
    """

    # Check if data_type is valid
    if not datatype.lower() in ["edx", "moke", "xrd"]:
        print("data_type must be one of 'EDX', 'MOKE', 'XRD'.")
        return 1
    else:
        datatypes = [datatype]

    dataset = xr.Dataset()

    print("Reading", datatype)
    positions = get_all_positions(hdf5_file, data_type=datatype)
    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    # Add measurement data
    for x, y in positions:
        if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
            continue
        measurement, units = search_measurement_data_from_type(
            hdf5_file, datatype, x, y
        )

        # Check if mutilples measurements of the same type are in the file
        keys = measurement.keys()
        if len(keys) > 1 and meas_key is None:
            print(
                "Multiple measurements found for ",
                datatype,
                ".\nPlease select one by passing one of the following keys in the meas_key argument:",
            )
            for key in keys:
                print(key)
            return 1
        else:
            if len(keys) == 1:
                meas_key = list(keys)[0]

        # Add measurement data
        add_measurement_data(
            dataset,
            measurement[meas_key],
            datatype,
            x,
            y,
            x_vals,
            y_vals,
            meas_key,
        )

        # Add units for x, y positions for all datasets
        position_units = get_position_units(hdf5_file, data_type=datatype)
        dataset["x"].attrs["units"] = position_units["x_pos"]
        dataset["y"].attrs["units"] = position_units["y_pos"]

        # Add units for scan axis in all datasets
        for key in units.keys():
            if datatype.lower() != "moke":
                if key in dataset:
                    dataset[key].attrs["units"] = units[key]

    return dataset


# def get_xrd_images(hdf5_file, exclude_wafer_edges=True):
#     dataset = xr.Dataset()

#     positions = _get_all_positions(hdf5_file, data_type="xrd")
#     x_vals = sorted(set([pos[0] for pos in positions]))
#     y_vals = sorted(set([pos[1] for pos in positions]))

#     for x, y, nb_scan in tqdm(positions):
#         if np.abs(x) + np.abs(y) > 60 and exclude_wafer_edges:
#             continue

#         group_path = make_group_path(["XRD", nb_scan, "image"])
#         image = get_xrd_image(hdf5_file, group_path)["2D_Camera_Image"]

#         if "image" not in dataset:
#             dataset["image"] = xr.DataArray(
#                 np.nan,
#                 coords=[
#                     y_vals,
#                     x_vals,
#                     np.arange(image.shape[0]),
#                     np.arange(image.shape[1]),
#                 ],
#                 dims=["y", "x", "pixel x", "pixel y"],
#             )
#         dataset["image"].loc[{"y": y, "x": x}] = image

#     return dataset


# def create_simplified_dataset(hdf5_file, hdf5_save_file):
#     """
#     Creates a simplified HDF5 dataset with the measurement data sorted by x and y position coordinates.

#     Parameters
#     ----------
#     hdf5_file : str or pathlib.Path
#         The path to the input HDF5 file.
#     hdf5_save_file : str or pathlib.Path
#         The path to the output HDF5 file.
#     """

#     group_list = ["edx", "moke", "xrd"]
#     coord_list = [
#         "({:.1f},{:.1f})".format(float(x), float(y))
#         for x in range(-40, 45, 5)
#         for y in range(-40, 45, 5)
#     ]

#     with h5py.File(hdf5_file, "r") as h5f, h5py.File(hdf5_save_file, "w") as h5f_save:

#         for group in h5f["./"]:
#             try:
#                 datatype = h5f[f"{group}"].attrs["HT_type"]
#             except KeyError:
#                 continue

#             if datatype in group_list:
#                 for coord in coord_list:
#                     # Check if the group already exists
#                     saved_group_coord = f"{group}/{coord}"

#                     if coord not in h5f_save:
#                         try:
#                             instrument = h5f[f"{saved_group_coord}"]["instrument"]
#                         except KeyError:
#                             continue

#                         h5f_save.create_group(f"{coord}")

#                         # Create x and y position datasets
#                         h5f_save[f"{coord}"].create_dataset(
#                             "x_pos", data=instrument["x_pos"]
#                         )
#                         h5f_save[f"{coord}"].create_dataset(
#                             "y_pos", data=instrument["y_pos"]
#                         )
#                         h5f_save[f"{coord}"]["x_pos"].attrs["units"] = instrument[
#                             "x_pos"
#                         ].attrs["units"]
#                         h5f_save[f"{coord}"]["y_pos"].attrs["units"] = instrument[
#                             "y_pos"
#                         ].attrs["units"]
#                         h5f_save[f"{coord}"]["x_pos"].attrs["HT_type"] = "position"
#                         h5f_save[f"{coord}"]["y_pos"].attrs["HT_type"] = "position"

#                     if coord not in h5f[f"{group}"].keys():
#                         # Giving NaN values for missing data
#                         node = h5f_save[f"{coord}"]
#                         results = h5f[f"{group}/(0.0,0.0)"]["results"]
#                         # If EDX (but should never happened)
#                         if datatype == "edx":
#                             for key in results.keys():
#                                 if "Element" in key:
#                                     node.create_dataset(key.split(" ")[-1], data=np.nan)
#                                     node[key.split(" ")[-1]].attrs["units"] = "at.%"
#                                     node[key.split(" ")[-1]].attrs["HT_type"] = datatype
#                         # If MOKE
#                         elif datatype == "moke":
#                             for key in results.keys():
#                                 if key == "coercivity_m0":
#                                     node.create_dataset(key, data=np.nan)
#                                     node[key.split(" ")[-1]].attrs[
#                                         "units"
#                                     ] = "Tesla (T)"
#                                     node[key.split(" ")[-1]].attrs["HT_type"] = datatype
#                                 elif key == "max_kerr_rotation":
#                                     pass
#                                     """node.create_dataset(key, data=np.nan)
#                                     node[key.split(" ")[-1]].attrs["HT_type"] = datatype"""

#                         # If XRD
#                         elif datatype == "xrd":
#                             saving_result_list = ["A", "B", "C", "phase_fraction"]

#                             for phase in results["phases"].keys():
#                                 for saving_key in saving_result_list:
#                                     if saving_key in results["phases"][phase].keys():
#                                         node.create_dataset(
#                                             f"{phase}_{saving_key}", data=np.nan
#                                         )
#                         continue

#                     # Creates new dataset with current datatype
#                     if datatype == "edx":
#                         node = h5f_save[f"{coord}"]
#                         results = h5f[f"{group}/{coord}"]["results"]
#                         for key in results.keys():
#                             if "Element" in key:
#                                 try:
#                                     node.create_dataset(
#                                         key.split(" ")[-1],
#                                         data=results[key]["AtomPercent"][()],
#                                     )
#                                     node[key.split(" ")[-1]].attrs["units"] = results[
#                                         key
#                                     ]["AtomPercent"].attrs["units"]
#                                     node[key.split(" ")[-1]].attrs["HT_type"] = datatype
#                                 except KeyError:
#                                     reference_results = h5f[f"{group}/(0.0,0.0)"][
#                                         "results"
#                                     ]
#                                     if (
#                                         key in reference_results.keys()
#                                         and "AtomPercent"
#                                         in reference_results[key].keys()
#                                     ):
#                                         node.create_dataset(
#                                             key.split(" ")[-1],
#                                             data=np.nan,
#                                         )
#                                         node[key.split(" ")[-1]].attrs["units"] = (
#                                             reference_results[key]["AtomPercent"].attrs[
#                                                 "units"
#                                             ]
#                                         )
#                                         node[key.split(" ")[-1]].attrs[
#                                             "HT_type"
#                                         ] = datatype

#                     elif datatype == "moke":
#                         node = h5f_save[f"{coord}"]
#                         results = h5f[f"{group}/{coord}"]["results"]
#                         for key in results.keys():
#                             if key == "coercivity_m0":
#                                 node.create_dataset(
#                                     key,
#                                     data=results[key]["mean"][()],
#                                 )
#                                 node[key].attrs["units"] = "Tesla (T)"
#                                 node[key].attrs["HT_type"] = datatype
#                             elif key == "max_kerr_rotation":
#                                 pass
#                                 """ node.create_dataset(
#                                     key,
#                                     data=results[key][()],
#                                 )
#                                 node[key].attrs["units"] = "Degrees (°)"
#                                 node[key].attrs["HT_type"] = datatype """

#                     elif datatype == "xrd":
#                         saving_result_list = ["A", "B", "C", "phase_fraction"]
#                         node = h5f_save[f"{coord}"]
#                         results = h5f[f"{group}/{coord}"]["results/phases"]
#                         measurement = h5f[f"{group}/{coord}"]["measurement"]

#                         # Fetching the results
#                         for phase in results.keys():
#                             for result in saving_result_list:
#                                 if result in results[phase].keys():

#                                     node.create_dataset(
#                                         f"{phase}_{result}",
#                                         data=(
#                                             str(results[phase][result][()])
#                                             .strip()
#                                             .split("+-")[0]
#                                         ),
#                                     )
#                                     try:
#                                         node[f"{phase}_{result}"].attrs["units"] = (
#                                             results[phase][result].attrs["units"]
#                                         )
#                                         node[f"{phase}_{result}"].attrs[
#                                             "HT_type"
#                                         ] = datatype
#                                     except KeyError:
#                                         # Taking into account missing attributes
#                                         pass

#                         # Fetching integrated intensity
#                         node.create_dataset(
#                             "CdTe_integrate_intensity",
#                             data=measurement["CdTe_integrate/intensity"][()],
#                         )
#                         node["CdTe_integrate_intensity"].attrs[
#                             "units"
#                         ] = "arbitrary unit (a.u.)"
#                         node["CdTe_integrate_intensity"].attrs["HT_type"] = datatype

#                         # Fetching integrated q
#                         node.create_dataset(
#                             "CdTe_integrate_q",
#                             data=measurement["CdTe_integrate/q"][()],
#                         )
#                         node["CdTe_integrate_q"].attrs["units"] = "Angstrom^-1 (A^-1)"
#                         node["CdTe_integrate_q"].attrs["HT_type"] = datatype

#                         # Fetching CdTe image
#                         node.create_dataset(
#                             "CdTe",
#                             data=measurement["CdTe"][()],
#                         )
#                         node["CdTe"].attrs["HT_type"] = datatype
