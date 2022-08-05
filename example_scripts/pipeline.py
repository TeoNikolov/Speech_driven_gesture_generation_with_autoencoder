import os
import subprocess
import argparse
import shutil

# General
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORK_DIR = os.path.join(SCRIPT_DIR, "..")
WORK_DIR_ORIGINAL = os.getcwd()
os.chdir(WORK_DIR)

# Argument parsing
parser = argparse.ArgumentParser()

# General params
parser.add_argument("--dataset",				required=True,				help="The path to the dataset folder.")
parser.add_argument("--model_dir", "-mdir", 								help="Directory where pre-trained models are saved to and loaded from.", default=os.path.join(WORK_DIR, "models"))
parser.add_argument("--train_ae", "-ae", 		action='store_true',		help="Train the autoencoder.")
parser.add_argument("--train_gg", "-gg", 		action='store_true',		help="Train the gesture generation model.")
parser.add_argument("--predict", "-pred", 		action='store_true',		help="Make predictions for all files in dataset folder.")

# Autoencoder params
parser.add_argument("--ae_dim", "-adim", 		type=int, 	default=8,		help="The autoencoder latent dimension.")
parser.add_argument("--ae_epochs", "-aeps", 	type=int,	default=5,		help="Train the autoencoder for this many epochs.")

# Gesture generation params
parser.add_argument("--gg_dim", "-gdim", 		type=int,	default=8,		help="The gesture generation model middle layer dimension.")
parser.add_argument("--gg_epochs", "-geps", 	type=int,	default=5,		help="Train the gesture generation model for this many epochs.")
parser.add_argument("--gg_period", "-period", 	type=int,	default=2, 		help="Save the gesture generation model after every X epochs.")
parser.add_argument("--gg_name", "-gname", 		type=str,	default="gg_model.hdf5", help="The name of the model. It is generally identifiable by the directory it is in.")

# Predict parameters
parser.add_argument("--predict_in", 										help="The filename of a .wav file to predict a gesture for. Used only with -pred flag.")
parser.add_argument("--predict_out", 										help="The filename of the output .bvh file that was predicted. Used only with -pred flag.")

# Post-processing parameters
parser.add_argument('--smoothing_mode', 		type=int, 	default=1,		help='How to apply smoothing to the produced gesture motion. (0) no smoothing; (1) Savitzky–Golay')
parser.add_argument('--savgol_window_length', 	type=int, 	default=13, 	help='If "--smoothing_mode = 1", specifies the window length (number of adjacent data samples) used by the Savitzky–Golay filter. Must be an odd number.')
parser.add_argument('--savgol_poly_order', 		type=int, 	default=3, 		help='If "--smoothing_mode = 1", specifies the polynomial order of the polynomial being fitted to the data by the Savitzky–Golay filter.')

args = parser.parse_args()

# short arg validation
if args.predict and not args.predict_in:
	raise ValueError("You requested to make predictions, but failed to provide an input .wav filename.")
if args.predict and not args.predict_out:
	raise ValueError("You requested to make predictions, but failed to provide an output .bvh filename.")
if args.smoothing_mode == 1:
	# summer school-specific validation
	if args.savgol_window_length < 3:
		raise ValueError("The parameter '--savgol_window_length' should not be less than 3!") # this is a lower boundary for the window size
	if args.savgol_window_length > 127:
		raise ValueError("The parameter '--savgol_window_length' should not be more than 127!") # technically it can be more than 127, but the result is already over-smoothed, so a boundary is used to point the user to a suitable range

	if args.savgol_poly_order < 1:
		raise ValueError("The parameter '--savgol_poly_order' cannot be less than 1!") # no really, the universe will collapse if it's set to 0
	if args.savgol_poly_order > 11:
		raise ValueError("The parameter '--savgol_poly_order' should not be more than 11!") # technically it can be more than 7, but the result is already similar to non-smoothed motion, so a boundary is used to point the user to a suitable range

	# general validation
	if args.savgol_window_length % 2 == 0:
		raise ValueError("The parameter '--savgol_window_length' must be an odd number!")
	if args.savgol_window_length < 3:
		raise ValueError("The parameter '--savgol_window_length' must be at least 3!")
	if args.savgol_poly_order >= args.savgol_window_length:
		raise ValueError("The parameter '--savgol_poly_order' must be less than '--savgol_window_length'!")





# Dataset
DATASET_NAME = os.path.basename(args.dataset)
DATASET_DIR = args.dataset # used mainly during training, but also during prediction for locating the AE and GG models
PREDICT_IN = args.predict_in
PREDICT_OUT = args.predict_out

# Autoencoder / Motion Represntation Learning
AE_EPOCHS = args.ae_epochs
AE_DIM = args.ae_dim
AE_TRAIN_SCRIPT = os.path.join(WORK_DIR, "motion_repr_learning", "ae", "learn_ae_n_encode_dataset.py")
AE_MODEL_DIR = os.path.join(args.model_dir, f"ae_{DATASET_NAME}_{AE_DIM}")
AE_CHKPT_DIR = os.path.join(AE_MODEL_DIR, "checkpoints")
AE_SMMRY_DIR = os.path.join(AE_MODEL_DIR, "summaries")

GG_EPOCHS = args.gg_epochs
GG_HIDDEN_DIM = args.gg_dim
GG_PERIOD = args.gg_period
GG_FILENAME = os.path.join(args.model_dir, f"gg_{DATASET_NAME}_{AE_DIM}_{GG_HIDDEN_DIM}", args.gg_name)
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
	subprocess.run(f'python {script_loc} --data_dir "{os.path.join(dataset_dir, "processed")}" --data_info_dir {dataset_dir} --chkpt_dir "{checkpoint_dir}" --summary_dir "{summary_dir}" --layer2_width {dim} --training_epochs {epochs}', shell=True)

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
	subprocess.run(f'python {tmplt["decode_script"]} -input_file {f_out_encoded} -output_file {f_out_decoded} --data_dir {data_dir} --data_info_dir {tmplt["dataset_dir"]} --chkpt_dir {tmplt["ae_chkpt_dir"]} --summary_dir {tmplt["ae_smmry_dir"]} --layer2_width {tmplt["ae_dim"]} --batch_size=8 --smoothing_mode {tmplt["smoothing_mode"]} --savgol_window_length {tmplt["savgol_window_length"]} --savgol_poly_order {tmplt["savgol_poly_order"]}', shell=True)

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
	'overwrite': False,
	'smoothing_mode': args.smoothing_mode,
	'savgol_window_length': args.savgol_window_length,
	'savgol_poly_order': args.savgol_poly_order
}

if args.predict:
	if not os.path.exists(os.path.dirname(PREDICT_OUT)):
		os.makedirs(os.path.dirname(PREDICT_OUT))
	predict(predict_template)
	
os.chdir(WORK_DIR_ORIGINAL)