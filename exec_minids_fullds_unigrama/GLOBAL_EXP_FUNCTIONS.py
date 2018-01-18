import os.path
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def mark_as_done(network_name_path):
	with open(network_name_path, 'a') as file:
		file.write('done!');

def is_executed(network_name_path):
	return os.path.isfile(network_name_path)

def extract_name(str):
	return str[0].split('.')[0]


def get_ae_callbacks(network_name, checkpoints_dir, tensorflow_dir):
	ae_callbacks = [
		EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
		ModelCheckpoint(checkpoints_dir + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=tensorflow_dir + network_name , histogram_freq=1, write_graph=True)	
	]
	return ae_callbacks

def get_mlp_callbacks(network_name, checkpoints_dir, tensorflow_dir):

	mlp_callbacks = [
		EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
		ModelCheckpoint(checkpoints_dir + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=tensorflow_dir + network_name + '_mlp', histogram_freq=1, write_graph=True)	
	]
	return mlp_callbacks