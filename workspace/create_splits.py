import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


"""
NOTE: This is the way to run this module: 

    python create_splits.py --data_dir data/waymo/
"""

def split(data_dir):
    """
    Description: This function picks the tfrecord files contained in data_dir/training_and_validation
    and divides the data into two segments - validation and training segments.

    NOTE: data/waymo/test folder already contains 3 tfrecord files before calling this module.
    that is why I did not send data into test folder.

    The data is stored as symbolic links instead of copy. To save space!

    The train data is stored in data_dir/train folder.
    The validation data is stored in data_dir/val folder.

    Params: data_dir - string - path to folders where the source data is contained in a folder called "training_and_validation"
    Returns: None
    """
    
    data_dir = os.path.abspath(data_dir)
    dataset_dir = os.path.join(data_dir, "training_and_validation")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    #this is an array of filenames only without path
    source_list = os.listdir(dataset_dir)
    #print(source_list)

    split_index = int(0.8 * len(source_list))

    train_set, val_set = np.split(source_list, [split_index])

    for filename in train_set:
        source_file = os.path.join(dataset_dir, filename)
        dest_file = os.path.join(train_dir, filename)
        #print(source_file)
        os.symlink(source_file, dest_file)
    
    for filename in val_set:
        source_file = os.path.join(dataset_dir, filename)
        dest_file = os.path.join(val_dir, filename)
        #print(source_file)
        os.symlink(source_file, dest_file)

    #print(dataset_dir)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)