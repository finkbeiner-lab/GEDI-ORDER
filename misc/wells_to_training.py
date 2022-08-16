"""Shuffle data from wells in different experiments to training, validation, testing"""

import shutil
import os
from glob import glob
import pandas as pd


def get_data_dict(platemaps, classes):
    # each plate is a different class
    data = {}
    for platemap in platemaps:
        df = pd.read_csv(platemap)
        for cls in classes:
            if cls not in data.keys():
                data[cls] = []
                # append plate and well
                for i, row in df.iterrows():
                    well = row['Sci_WellID']
                    cond = row['Condition']
                    plate = row['Sci_PlateID']
                    if cond == cls:
                        data[cls].append((plate, well))

    return data


def shuffle_data(plate_dirs: list, datadict: dict):
    class_dict = {}
    for cls, plate_well in datadict.items():
        if cls not in class_dict.keys():
            class_dict[cls] = []
        for plate, well in plate_well:
            for plate_dir in plate_dirs:  #[/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau]
                if plate in plate_dir:   # P301S-Tau in /mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau
                    use_plate = plate_dir  # todo: assumes all substrings are unique in filepaths
                    break
            class_dict[cls] += glob(os.path.join(use_plate, well, '*.tif'))   # /mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau/A3/*.tif

    train, val, test = [],[],[]
    split = [.7., .15, .15]



