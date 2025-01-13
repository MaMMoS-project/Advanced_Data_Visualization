# -*- coding: utf-8 -*-
"""
to complete

@author: williamrigaut
"""

import h5py
import numpy as np
from packages.compilers.compile_hdf5 import convertFloat


def writeHDF5_MOKE(HDF5_path, MOKE_path, filename, x_pos, y_pos):
    header_path = f"{MOKE_path}/info.txt"
    header_MOKE = [["Sample name"], ["Date"]]
    sample_name = (MOKE_path.split("/"))[-1]
    result_path = f"./results/MOKE/{sample_name}_MOKE.dat"
    data_mag = filename
    data_pul = filename.replace("magnetization", "pulse")
    data_sum = filename.replace("magnetization", "sum")

    data_magnetization = []
    data_pulse = []
    data_reflectivity = []
    results_MOKE = []

    with open(header_path, "r", encoding="iso-8859-1") as header:
        for j, line in enumerate(header):
            if j == 0 or j == 1:
                header_MOKE[j].append(line.split("#")[-1].strip())
            else:
                header_MOKE.append(line.strip().split("="))
    nb_aq = int(header_MOKE[-1][1])

    with open(data_mag, "r", encoding="iso-8859-1") as magnetization, open(
        data_pul, "r", encoding="iso-8859-1"
    ) as pulse, open(data_sum, "r", encoding="iso-8859-1") as reflectivity:

        data_magnetization = readMokeDataFile(magnetization)
        data_pulse = readMokeDataFile(pulse)
        data_reflectivity = readMokeDataFile(reflectivity)

    with open(result_path, "r", encoding="iso-8859-1") as file:
        results_header = next(file).split("\t")
        for line in file:
            line_values = [round(float(elm), 3) for elm in line.split()]
            if int(line_values[0]) == x_pos and int(line_values[1]) == y_pos:
                results_MOKE = [
                    (results_header[i + 2].strip(), elm)
                    for i, elm in enumerate(line_values[2:])
                ]

    with h5py.File(HDF5_path, "a") as f:
        file_group_path = f"MOKE/Scan_({x_pos}, {y_pos})"
        f.create_group(file_group_path)

        data = f.create_group(f"{file_group_path}/Data")
        moke_header = f.create_group(f"{file_group_path}/Header")
        moke_results = f.create_group(f"{file_group_path}/Results")

        for elm in header_MOKE:
            elm_1 = convertFloat(elm[1].strip())
            moke_header.create_dataset(elm[0], (1,), data=elm_1)
        for elm in results_MOKE:
            moke_results.create_dataset(elm[0], (1,), data=elm[1])

        mag_dset = data.create_dataset(
            "Magnetization", (len(data_magnetization), nb_aq), data=data_magnetization
        )
        pul_dset = data.create_dataset(
            "Pulse", (len(data_pulse), nb_aq), data=data_pulse
        )
        sum_dset = data.create_dataset(
            "Reflectivity", (len(data_reflectivity), nb_aq), data=data_reflectivity
        )
