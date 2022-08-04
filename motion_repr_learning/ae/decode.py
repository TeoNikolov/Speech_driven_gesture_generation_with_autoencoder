"""
This file contains a usage script, intended to test using interface.
Developed by Taras Kucherenko (tarask@kth.se)
"""
import sys
sys.path.append('.')
import numpy as np

import train as tr
from learn_ae_n_encode_dataset import create_nn, prepare_motion_data
from config import args

import numpy as np

import sys

from scipy.signal import savgol_filter


def smoothing(motion, window_length, poly_order):

    smoothed = [savgol_filter(motion[:,i], window_length, poly_order) for i in range(motion.shape[1])]

    new_motion = np.array(smoothed).transpose()

    return new_motion


if __name__ == '__main__':
    # Make sure that the two mandatory arguments are provided
    if args.input_file is None or args.output_file is None:
        print("Usage: python decode.py -input_file INPUT_FILE -output_file OUTPUT_FILE \n" + \
              "Where INPUT_FILE is the encoded prediction file and OUTPUT_FILE is the file in which the decoded gestures will be saved.")
        exit(-1)
    
    # For decoding these arguments are always False and True
    args.pretrain_network = False
    args.load_model_from_checkpoint = True

    # Train the network
    nn = create_nn(None, None, None, None, just_inference=True)

    # Read the encoding
    encoding = np.load(args.input_file)

    print(encoding.shape)

    # Decode it
    decoding = tr.decode(nn, encoding)

    # Smoothing
    if args.smoothing_mode == 1:
        print(f"Smoothing using Savitzky–Golay digital filter with window length {args.savgol_window_length} and poly order {args.savgol_poly_order}...")
        decoding = smoothing(decoding, args.savgol_window_length, args.savgol_poly_order)

    print(decoding.shape)

    np.save(args.output_file, decoding)

    # Close Tf session
    nn.session.close()
