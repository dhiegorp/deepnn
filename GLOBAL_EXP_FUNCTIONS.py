import os.path
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

def mark_as_done(network_name_path):
	with open(network_name_path, 'a') as file:
		file.write('done!');

def is_executed(network_name_path):
	return os.path.isfile(network_name_path)

def extract_name(str):
	return str[0].split('.')[0]



def get_ae_callbacks(network_name):
	ae_callbacks = [
		EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		#TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=0, write_graph=False)	
		CSVLogger(GLOBAL['reports_dir'] + network_name + '.csv')
	]
	return ae_callbacks

def get_mlp_callbacks(network_name):

	mlp_callbacks = [
		EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		#TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=0, write_graph=False)	
		CSVLogger(GLOBAL['reports_dir'] + network_name + '_mlp.csv')
	]
	return mlp_callbacks
