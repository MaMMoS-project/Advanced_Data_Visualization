# -*- coding: utf-8 -*-
"""
to complete

@author: williamrigaut
"""

import h5py
import numpy as np
from packages.compilers.compile_hdf5 import convertFloat


def get_scan_number(filepath):
    scan_number = filepath.name.split("_")[0].replace("p", "")

    return scan_number


def get_wafer_positions(filepath):
    x_pos = filepath.name.split("_")[1].split("x")[-1]
    y_pos = filepath.name.split("_")[2].split("y")[-1]

    return x_pos, y_pos


def read_header_from_moke(filepath):
    header_dict = {}
    fullpath = filepath.parent / "info.txt"

    with open(fullpath, "r", encoding="iso-8859-1") as file:
        header_dict["Sample name"] = file.readline().strip().replace("#", "")
        header_dict["Date"] = file.readline().strip().replace("#", "")
        for line in file:
            key, value = line.strip().split("=")
            header_dict[key] = value

    return header_dict


def read_data_from_moke(filepath):
    mag_data, pul_data, sum_data = [], [], []

    mag_path = filepath
    pul_path = filepath.parent / f"{filepath.name.replace('magnetization', 'pulse')}"
    sum_path = filepath.parent / f"{filepath.name.replace('magnetization', 'sum')}"

    with open(mag_path, "r") as magnetization, open(pul_path, "r") as pulse, open(
        sum_path, "r"
    ) as reflectivity:

        magnetization = magnetization.readlines()
        pulse = pulse.readlines()
        reflectivity = reflectivity.readlines()

        for mag, pul, sum in zip(magnetization[2:], pulse[2:], reflectivity[2:]):
            mag = mag.strip().split()
            pul = pul.strip().split()
            sum = sum.strip().split()

            mag_data.append([float(elm) for elm in mag])
            pul_data.append([float(elm) for elm in pul])
            sum_data.append([float(elm) for elm in sum])

    return mag_data, pul_data, sum_data


def get_time_from_moke(datasize):
    time_step = 0.05  # in microsecondes (or 50ns)
    time = [j * time_step for j in range(datasize)]

    return time


def get_results_from_moke(filepath):
    results_dict = {}

    return results_dict


def set_instrument_from_dict(moke_dict, node):
    for key, value in moke_dict.items():
        if isinstance(value, dict):
            set_instrument_from_dict(value, node.create_group(key))
        else:
            node[key] = value

    return None


def write_moke_to_hdf5(HDF5_path, filepath, mode="a"):
    scan_number = get_scan_number(filepath)
    x_pos, y_pos = get_wafer_positions(filepath)

    header_dict = read_header_from_moke(filepath)
    mag_dict, pul_dict, sum_dict = read_data_from_moke(filepath)
    time_dict = get_time_from_moke(len(mag_dict))
    nb_aquisitions = len(mag_dict[0])

    results_dict = get_results_from_moke(filepath)

    with h5py.File(HDF5_path, mode) as f:
        scan_group = f"/entry/moke/scan_{scan_number}/"
        scan = f.create_group(scan_group)

        # Instrument group for metadata
        instrument = scan.create_group("instrument")
        instrument.attrs["NX_class"] = "HTinstrument"
        instrument.attrs["x_pos"] = convertFloat(x_pos)
        instrument.attrs["y_pos"] = convertFloat(y_pos)

        set_instrument_from_dict(header_dict, instrument)

        # Data group
        data = scan.create_group("measurement")
        data.attrs["NX_class"] = "HTmeasurement"
        time = [convertFloat(t) for t in time_dict]
        time_node = data.create_dataset("time", data=time, dtype="float")
        time_node.attrs["units"] = "μm"

        for i in range(nb_aquisitions):
            mag = [convertFloat(t[i]) for t in mag_dict]
            mag_node = data.create_dataset(
                f"magnetization_{i+1}", data=mag, dtype="float"
            )

            pul = [convertFloat(t[i]) for t in pul_dict]
            pul_node = data.create_dataset(f"pulse_{i+1}", data=pul, dtype="float")

            sum = [convertFloat(t[i]) for t in sum_dict]
            sum_node = data.create_dataset(
                f"reflectivity_{i+1}", data=sum, dtype="float"
            )

            mag_node.attrs["units"] = "V"
            pul_node.attrs["units"] = "V"
            sum_node.attrs["units"] = "V"

        # Results group
        results = scan.create_group("results")
        results.attrs["NX_class"] = "HTresults"
        set_instrument_from_dict(results_dict, results)
