from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import os.path
from ENVIRONMENT import *

environment = Environment()
base_path = environment.base_path
ds_path = environment.dataset_base_path

GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	'log_dir': base_path + '/logs/10layers/unigram/',
	'reports_dir': base_path + '/reports/10layers/unigram/',
	'fullds_reports_dir': base_path + '/reports/10layers/unigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/10layers/unigram/',
	'checkpoints_dir':base_path + '/checkpoints/10layers/unigram/',
	'executed_path':base_path + '/executed/10layers/unigram/',
	'data_dir': ds_path + '/',
	'fullds_data_dir': ds_path + '/',
	#'data_dir': ds_path + '/malware_selected_1gram_mini.pkl',
	#'fullds_data_dir':ds_path + '/malware_selected_1gram.pkl',
	

	'data_target_list' : [1,2,3,4,5,6,7,8,9],
	'epochs': 1000,
	'batch': 32,
	'store_history' : True,
	'shuffle_batches' : True,	
	'autoencoder_configs' : {
		'hidden_layer_activation' : 'relu',
		'output_layer_activation' : 'relu',
		'loss_function' : 'mse',
		'optimizer': SGD(lr=0.01),
		'discard_decoder_function': True
	},
	'mlp_configs': {
		'activation' : 'sigmoid',
		#'activation' : 'softmax',
		'loss_function' : 'categorical_crossentropy',
		'optimizer' : SGD(lr=0.01),
		'use_last_dim_as_classifier' : False,
		'classifier_dim' : 9
	}


}

  
MAP_DIMS = {
	#'AE_UNIGRAMA_4L_9FULLDS_UNDER_01' :  [96,  28,  26, 24, 22, 9],
	#'AE_UNIGRAMA_4L_9FULLDS_UNDER_02' :  [96,  76,  69, 63, 56, 9],
	#'AE_UNIGRAMA_4L_9FULLDS_UNDER_03':   [96,  86,  78, 71, 63, 9],
	#'AE_UNIGRAMA_4L_9FULLDS_OVER_04':    [96, 134, 122, 109, 97, 9],
	#'AE_UNIGRAMA_4L_9FULLDS_OVER_05' :   [96, 172, 156, 139, 123, 9],
	#'AE_UNIGRAMA_4L_FULLDS_UNDER_01' :  [96,  28,  26, 24, 22],
	#'AE_UNIGRAMA_4L_FULLDS_UNDER_02' :  [96,  76,  69, 63, 56],
	#'AE_UNIGRAMA_4L_FULLDS_UNDER_03':   [96,  86,  78, 71, 63],
	#'AE_UNIGRAMA_4L_FULLDS_OVER_04':    [96, 134, 122, 109, 97],
	#'AE_UNIGRAMA_4L_FULLDS_OVER_05' :   [96, 172, 156, 139, 123],
	#'AE_UNIGRAMA_4L_MINIDS_UNDER_01' :  [96,  28,  26, 24, 22],
	#'AE_UNIGRAMA_4L_MINIDS_UNDER_02' :  [96,  76,  69, 63, 56],
	#'AE_UNIGRAMA_4L_MINIDS_UNDER_03':   [96,  86,  78, 71, 63],
	#'AE_UNIGRAMA_4L_MINIDS_OVER_04':    [96, 134, 122, 109, 97],
	#'AE_UNIGRAMA_4L_MINIDS_OVER_05' :   [96, 172, 156, 139, 123]

	'AE_UNIGRAMA_10L_FULLDS_OVER_01' :  [96, 144, 130, 117, 103, 90 , 76 , 63, 49, 36, 22],
	'AE_UNIGRAMA_10L_FULLDS_OVER_02' :  [96, 134, 121, 109, 96 , 84 , 71 , 59, 46, 34, 21],
	'AE_UNIGRAMA_10L_FULLDS_UNDER_03' : [96, 19 , 18 , 17 , 16 , 15 , 14 , 13, 12, 11, 10],
	'AE_UNIGRAMA_10L_FULLDS_OVER_04' :  [96, 163, 148, 132, 117, 101, 86 , 71, 55, 40, 24],
	'AE_UNIGRAMA_10L_FULLDS_OVER_05' :  [96, 192, 174, 155, 137, 119, 100, 82, 64, 46, 27]
}

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
