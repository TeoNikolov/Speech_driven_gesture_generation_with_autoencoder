import os
import subprocess
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--train_ae", "-ae", action='store_true')
parser.add_argument("--train_gg", "-gg", action='store_true')
parser.add_argument("--predict", "-pred", action='store_true')
args = parser.parse_args()

# General
SCRIPT_DIR = os.getcwd()
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, 'pipeline.py')
DATASET = os.path.join(SCRIPT_DIR, "..", "..", "..", "dataset", "10-percent")
PREDICT_DIR_IN = os.path.join(SCRIPT_DIR, "..", "..", "..", "dataset", "my_audio")
PREDICT_DIR_OUT_BASE = os.path.join(SCRIPT_DIR, "..", "..", "..", "dataset", "taras_bl_output")

# train autoencoders
AE_EPOCHS = 3
AE_DIMS = [8, 32, 256, 1024]
if args.train_ae:
    for adim in AE_DIMS:
        subprocess.run(f'python {PIPELINE_SCRIPT} --dataset {DATASET} -adim {adim} -aeps {AE_EPOCHS} -ae')

# train the gesture generators
GG_EPOCHS = 4
GG_DIMS = [4, 32, 128]
GG_PERIOD = 10
if args.train_gg:
    for gdim in GG_DIMS:
        for adim in AE_DIMS:
            subprocess.run(f'python {PIPELINE_SCRIPT} --dataset {DATASET} -adim {adim} -aeps {AE_EPOCHS} -gdim {gdim} -geps {GG_EPOCHS} -period {GG_PERIOD} -gg')

# batch predict
# GG_DIMS=[4] #debug
# AE_DIMS=[8] #debug
if args.predict:
    for gdim in GG_DIMS:
        for adim in AE_DIMS:
            predict_dir_out = os.path.join(PREDICT_DIR_OUT_BASE, f'ad-{adim}_gd-{gdim}')
            for in_file in glob.glob(os.path.join(PREDICT_DIR_IN, "*.wav")):
                in_file = os.path.basename(in_file)
                p_in = os.path.join(PREDICT_DIR_IN, in_file)
                p_out = os.path.join(predict_dir_out, in_file.replace('.wav', '.bvh'))
                subprocess.run(f'python {PIPELINE_SCRIPT} --dataset {DATASET} --predict_in {p_in} --predict_out {p_out} -adim {adim} -gdim {gdim} -pred')
