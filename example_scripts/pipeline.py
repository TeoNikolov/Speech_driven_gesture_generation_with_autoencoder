import os
import subprocess
import argparse
import shutil

# General
SCRIPT_DIR = os.getcwd()
WORK_DIR = os.path.join(SCRIPT_DIR, "..")
os.chdir(WORK_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="The path to the dataset folder.", required=True)
parser.add_argument("--predict_in", help="The filename of a .wav file to predict a gesture for. Used only with -pred flag.")
parser.add_argument("--predict_out", help="The filename of the output .bvh file that was predicted. Used only with -pred flag.")
parser.add_argument("--train_ae", "-ae", help="Train the autoencoder.", action='store_true')
parser.add_argument("--train_gg", "-gg", help="Train the gesture generation model.", action='store_true')
parser.add_argument("--predict", "-pred", help="Make predictions for all files in dataset folder.", action='store_true')
parser.add_argument("--ae_dim", "-adim", help="The autoencoder latent dimension.", default=8, type=int)
parser.add_argument("--ae_epochs", "-aeps", help="Train the autoencoder for this many epochs.", default=5, type=int)
parser.add_argument("--gg_dim", "-gdim", help="The gesture generation model middle layer dimension.", default=8, type=int)
parser.add_argument("--gg_epochs", "-geps", help="Train the gesture generation model for this many epochs.", default=5, type=int)
parser.add_argument("--gg_period", "-period", help="Save the gesture generation model after every X epochs.", default=2, type=int)
parser.add_argument("--gg_name", "-gname", help="The name of the model. It is generally identifiable by the directory it is in.", default="gg_model.hdf5", type=str)
args = parser.parse_args()

# short arg validation
if args.predict and not args.predict_in:
	raise ValueError("You requested to make predictions, but failed to provide an input .wav filename.")
if args.predict and not args.predict_out:
	raise ValueError("You requested to make predictions, but failed to provide an output .bvh filename.")

# Dataset
DATASET_NAME = os.path.basename(args.dataset)
DATASET_DIR = args.dataset # used mainly during training, but also during prediction for locating the AE and GG models
PREDICT_IN = args.predict_in
PREDICT_OUT = args.predict_out

# Autoencoder / Motion Represntation Learning
AE_EPOCHS = args.ae_epochs
AE_DIM = args.ae_dim
AE_TRAIN_SCRIPT = os.path.join(WORK_DIR, "motion_repr_learning", "ae", "learn_ae_n_encode_dataset.py")
AE_MODEL_DIR = os.path.join(WORK_DIR, "models", f"ae_{DATASET_NAME}_{AE_DIM}")
AE_CHKPT_DIR = os.path.join(AE_MODEL_DIR, "checkpoints")
AE_SMMRY_DIR = os.path.join(AE_MODEL_DIR, "summaries")

GG_EPOCHS = args.gg_epochs
GG_HIDDEN_DIM = args.gg_dim
GG_PERIOD = args.gg_period
GG_FILENAME = os.path.join(WORK_DIR, "models", f"gg_{DATASET_NAME}_{AE_DIM}_{GG_HIDDEN_DIM}", args.gg_name)
GG_TRAIN_SCRIPT = os.path.join(WORK_DIR, "train.py")

ENCODE_SCRIPT = os.path.join(WORK_DIR, "data_processing", "encode_audio.py")
PREDICT_SCRIPT = os.path.join(WORK_DIR, "predict.py")
DECODE_SCRIPT = os.path.join(WORK_DIR, "motion_repr_learning", "ae", "decode.py")
FEAT2BVH_SCRIPT = os.path.join(WORK_DIR, "data_processing", "features2bvh.py")

def train_ae(script_loc, dataset_dir, model_dir, checkpoint_dir, summary_dir, dim, epochs):
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
		os.mkdir(checkpoint_dir)
		os.mkdir(summary_dir)
	subprocess.run(f'python {script_loc} --data_dir "{os.path.join(dataset_dir, "processed")}" --chkpt_dir "{checkpoint_dir}" --summary_dir "{summary_dir}" --layer2_width {dim} --training_epochs {epochs}', shell=True)

def train_gg(script_loc, dataset_dir, model_file, epochs, ae_dim, hidden_dim, save_period):
	subprocess.run(f'python {script_loc} {model_file} {epochs} {os.path.join(dataset_dir, "processed")} 26 True {ae_dim} -period {save_period} -dim {hidden_dim}', shell=True)

def predict(tmplt):
	if not tmplt["overwrite"] and os.path.exists(tmplt["file_out"]):
		return

	f_encoded = os.path.join(os.path.dirname(tmplt["file_out"]), os.path.basename(tmplt["file_in"].replace('.wav', '_encoded.npy')))
	f_out_encoded = f_encoded.replace('_encoded.npy', '_out_encoded.npy')
	f_out_decoded = f_encoded.replace('_encoded.npy', '_out_decoded.npy')

	shutil.copyfile(tmplt["file_in"], os.path.join(os.path.dirname(tmplt["file_out"]), os.path.basename(tmplt["file_in"])))

	os.chdir('data_processing')
	subprocess.run(f'python {tmplt["encode_script"]} -i {tmplt["file_in"]} -o {f_encoded}', shell=True)
	os.chdir('..')

	# python predict.py test-model-name.hdf5 dataset\processed\dev_inputs\X_dev_trn_2022_v1_001.npy output\X_dev_trn_2022_v1_001_out_encoded.npy
	subprocess.run(f'python {tmplt["predict_script"]} {tmplt["gg_filename"]} {f_encoded} {f_out_encoded}', shell=True)

	# python motion_repr_learning/ae/decode.py -input_file output\X_dev_trn_2022_v1_001_out_encoded.npy -output_file output\X_dev_trn_2022_v1_001_out_decoded.npy --layer1_width 128 --batch_size=8
	data_dir = os.path.join(tmplt["dataset_dir"], 'processed')
	subprocess.run(f'python {tmplt["decode_script"]} -input_file {f_out_encoded} -output_file {f_out_decoded} --data_dir {data_dir} --chkpt_dir {tmplt["ae_chkpt_dir"]} --summary_dir {tmplt["ae_smmry_dir"]} --layer2_width {tmplt["ae_dim"]} --batch_size=8', shell=True)

	# python features2bvh.py -feat ..\output\X_dev_trn_2022_v1_001_out_decoded.npy -bvh ..\output\X_dev_trn_2022_v1_001_out_decoded.bvh
	os.chdir('data_processing')
	subprocess.run(f'python {tmplt["feat2bvh_script"]} -feat {f_out_decoded} -bvh {tmplt["file_out"]} -pipe {tmplt["dataset_dir"] + "/"}', shell=True)
	os.chdir('..')
	
if args.train_ae:
	train_ae(AE_TRAIN_SCRIPT, DATASET_DIR, AE_MODEL_DIR, AE_CHKPT_DIR, AE_SMMRY_DIR, AE_DIM, AE_EPOCHS)

if args.train_gg:
	if not os.path.exists(os.path.dirname(GG_FILENAME)):
		os.makedirs(os.path.dirname(GG_FILENAME))
	train_gg(GG_TRAIN_SCRIPT, DATASET_DIR, GG_FILENAME, GG_EPOCHS, AE_DIM, GG_HIDDEN_DIM, GG_PERIOD)

predict_template = {
	'encode_script': ENCODE_SCRIPT,
	'decode_script': DECODE_SCRIPT,
	'predict_script': PREDICT_SCRIPT,
	'feat2bvh_script': FEAT2BVH_SCRIPT,
	'file_in': PREDICT_IN,
	'file_out': PREDICT_OUT,
	'gg_filename': GG_FILENAME,
	'dataset_dir': DATASET_DIR,
	'ae_chkpt_dir': AE_CHKPT_DIR,
	'ae_smmry_dir': AE_SMMRY_DIR,
	'ae_dim': AE_DIM,
	'overwrite': False
}

if args.predict:
	if not os.path.exists(os.path.dirname(PREDICT_OUT)):
		os.makedirs(os.path.dirname(PREDICT_OUT))
	predict(predict_template)