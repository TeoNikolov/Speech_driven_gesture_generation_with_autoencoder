"""
This script is used to split the dataset into train, test and dev sets.
More info on its usage is given in the main README.md file 

@authors: Taras Kucherenko, Rajmund Nagy
"""

import sys
import os
import shutil
import pandas
from os import path

# Params
from data_params import parser

# Indices for train/dev split
DEV_LAST_ID = 15
TRAIN_LAST_ID = 300

audio_prefix = motion_prefix = "trn_2022_v1_"


def copy_files(ind, raw_d_dir, processed_d_dir, data_split, suffix=""):
    # Copy audio
    filename = f"{audio_prefix}{ind}{suffix}.wav"
    original_file_path = path.join(raw_d_dir, "wav", filename)


    if os.path.isfile(original_file_path):
        target_file_path = path.join(processed_d_dir, data_split, "inputs", filename)
        shutil.copy(original_file_path, target_file_path)

    # Copy gestures
    filename = f"{motion_prefix}{ind}.npz"
    original_file_path = path.join(raw_d_dir, "bvh", filename)
    
    if os.path.isfile(original_file_path):
        target_file_path = path.join(processed_d_dir, data_split, "labels", filename)
        shutil.copy(original_file_path, target_file_path)      

def create_dataset_splits(raw_d_dir, processed_d_dir):
    """Create the train/dev/test splits in new subfolders within 'processed_d_dir'."""
    _create_data_directories(processed_d_dir)

    # prepare dev data
    for i in range(1, DEV_LAST_ID):
        copy_files(str(i).zfill(3), raw_d_dir, processed_d_dir, "dev")

    # prepare training data
    for i in range(DEV_LAST_ID, TRAIN_LAST_ID):
        copy_files(str(i).zfill(3), raw_d_dir, processed_d_dir, "train")
        copy_files(str(i).zfill(3), raw_d_dir, processed_d_dir, "train", "_2")

    extracted_dir = path.join(processed_d_dir)

    dev_files, train_files = _format_datasets(extracted_dir)

    # Save the filenames of each datapoints (the preprocessing script will use these)
    dev_files.to_csv(path.join(extracted_dir, "dev-dataset-info.csv"), index=False)
    train_files.to_csv(path.join(extracted_dir, "train-dataset-info.csv"), index=False)


def _create_data_directories(processed_d_dir):
    """Create subdirectories for the dataset splits."""
    dir_names = ["dev", "train"]
    sub_dir_names = ["inputs", "labels"]
    os.makedirs(processed_d_dir, exist_ok = True)
    
    print("Creating the datasets in the following directories:") 
    for dir_name in dir_names:
        dir_path = path.join(processed_d_dir, dir_name)
        print('  ', path.abspath(dir_path))
        os.makedirs(dir_path, exist_ok=True)  # e.g. ../../dataset/processed/train

        for sub_dir_name in sub_dir_names:
            dir_path = path.join(processed_d_dir, dir_name, sub_dir_name)
            os.makedirs(dir_path, exist_ok = True) # e.g. ../../dataset/processed/train/inputs/
    print()


def _format_datasets(extracted_dir):
    print("The datasets will contain the following indices:", end='')
    dev_files = _files_to_pandas_dataframe(extracted_dir, "dev", range(1, DEV_LAST_ID))
    train_files = _files_to_pandas_dataframe(extracted_dir, "train", range(DEV_LAST_ID, TRAIN_LAST_ID))
    print()

    return dev_files, train_files


def _files_to_pandas_dataframe(extracted_dir, set_name, idx_range):
    info_msg = f"\n  {set_name}:"
    print("{:10}".format(info_msg), end='')

    files = []
    for idx in idx_range:
        # original files
        input_file = path.abspath(path.join(extracted_dir, set_name, "inputs", audio_prefix + str(idx).zfill(3) + ".wav"))
        label_file = path.abspath(path.join(extracted_dir, set_name, "labels", motion_prefix + str(idx).zfill(3) + ".npz"))
        if os.path.isfile(input_file):
            files.append((input_file, label_file))

            print(idx, end=' ')

    return pandas.DataFrame(data=files, columns=["wav_filename", "bvh_filename"])


def check_dataset_directories(raw_data_dir):
    """
    Verify that 'raw_data_dir' exists and that it contains the 
    'speech' and 'motion' subdirectories.
    """
    if not path.isdir(raw_data_dir):
        abs_path = path.abspath(raw_data_dir)

        print(f"ERROR: The given dataset folder for the raw data ({abs_path}) does not exist!")
        print("Please, provide the correct path to the dataset in the `-raw_data_dir` argument.")
        exit(-1)

    speech_dir     = path.join(raw_data_dir, "wav")
    motion_dir     = path.join(raw_data_dir, "bvh")

    for sub_dir in [speech_dir, motion_dir]:
        if not path.isdir(sub_dir):
            _, name = path.split(sub_dir)
            print(f"ERROR: The '{name}' directory is missing from the given dataset folder: '{raw_data_dir}'!") 
            exit(-1)


if __name__ == "__main__":
    args = parser.parse_args()
    
    check_dataset_directories(args.raw_data_dir)
    create_dataset_splits(args.raw_data_dir, args.proc_data_dir)
    
    print(f"\nFinished!")
