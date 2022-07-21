"""
This is the main script for the training.
It contains speech-motion neural network implemented in Keras
This script should be used to train the model, as described in READ.me
"""

import sys
from os.path import join
import argparse
from xmlrpc.client import Boolean, boolean

import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_period", "-period", help="The frequency at which the model will be saved in the output folder.", default=5, type=int)
parser.add_argument("--hidden_dim", "-dim", help="The size of the hidden dimension of the middle layers.", default=256, type=int)
args, unknown_args = parser.parse_known_args()

# Check if script get enough parameters
if len(sys.argv) < 6:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE (DIM)')
ENCODED = sys.argv[5].lower() == 'true'

if ENCODED:
    if len(sys.argv) < 7:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE DIM')
    else:    
        N_OUTPUT = int(sys.argv[6])  # Representation dimensionality
else:
    N_OUTPUT = 498  # Number of Gesture Features


EPOCHS = int(sys.argv[2])
DATA_DIR = sys.argv[3]
N_INPUT = int(sys.argv[4])  # Number of input features

BATCH_SIZE = 1024
N_ENC = 64
N_HIDDEN = args.hidden_dim

N_CONTEXT = 60 + 1  # The number of frames in the context


def train(model_file):
    """
    Train a neural network to take speech as input and produce gesture as an output

    Args:
        model_file: file to store the model

    Returns:

    """

    # Get the data
    X = np.load(join(DATA_DIR, 'X_train.npy'))

    if ENCODED:

        # If we learn speech-representation mapping we use encoded motion as output
        Y = np.load(join(DATA_DIR, str(N_OUTPUT), 'Y_train_encoded.npy'))

        # Correct the sizes
        train_size = min(X.shape[0], Y.shape[0])
        X = X[:train_size]
        Y = Y[:train_size]

    else:
        Y = np.load(join(DATA_DIR, 'Y_train.npy'))

    N_train = int(len(X)*0.9)
    N_validation = len(X) - N_train

    # Split on training and validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)

    # Define Keras model

    model = Sequential()
    model.add(TimeDistributed(Dense(N_ENC), input_shape=(N_CONTEXT, N_INPUT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(GRU(N_HIDDEN, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    print(model.summary())

    optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model_checkpoint_cb = ModelCheckpoint(
        filepath=model_file.replace('.hdf5', '_epoch{epoch}.hdf5'),
        period=args.checkpoint_period
    )

    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, Y_validation), callbacks=[model_checkpoint_cb])
     
    model.save(model_file)

    # Save convergence results into an image
    pyplot.plot(hist.history['loss'], linewidth=3, label='train')
    pyplot.plot(hist.history['val_loss'], linewidth=3, label='valid')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.savefig(model_file.replace('hdf5', 'png'))


if __name__ == "__main__":
    model_file = sys.argv[1]
    
    if not model_file.endswith(".hdf5"):
        model_file += ".hdf5"
    
    train(model_file)
