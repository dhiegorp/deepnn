from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os.path
from ENVIRONMENT import *

environment = Environment()
base_path = environment.base_path
ds_path = environment.dataset_base_path    


GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	'log_dir': base_path + '/logs/1layer/bigram/',
	'reports_dir': base_path + '/reports/1layer/bigram/',
	'fullds_reports_dir': base_path + '/reports/1layer/bigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/1layer/bigram/',
	'checkpoints_dir':base_path + '/checkpoints/1layer/bigram/',
	'executed_path':base_path + '/executed/1layer/bigram/',
	'data_dir': ds_path + '/',
	'fullds_data_dir': ds_path + '/',
	#'data_dir': ds_path + '/malware_selected_1gram_mini.pkl',
	#'fullds_data_dir':ds_path + '/malware_selected_1gram.pkl',
	

	'data_target_list' : [1,2,3,4,5,6,7,8,9],
	#'epochs': 50,
	'epochs': 200,
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
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_1' : [9216, 922],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_2' : [9216, 1843],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_3' : [9216, 2765],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_4' : [9216, 3686],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_5' : [9216, 4608],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_6' : [9216, 5530],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_7' : [9216, 6451],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_8' : [9216, 7373],
	'AE_BIGRAMA_1L_MINIDS_UNDER_F0_9' : [9216, 8294],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_0' : [9216, 9216],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_1' : [9216, 10138],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_2' : [9216, 11059],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_3' : [9216, 11981],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_4' : [9216, 12902],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_5' : [9216, 13824],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_6' : [9216, 14746],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_7' : [9216, 15667],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_8' : [9216, 16589],
	'AE_BIGRAMA_1L_MINIDS_OVER_F1_9' : [9216, 17510],
	'AE_BIGRAMA_1L_MINIDS_OVER_F2_0' : [9216, 18432]
}

def get_ae_callbacks(network_name):
	ae_callbacks = [
		#EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=0, write_graph=False)	
	]
	return ae_callbacks

def get_mlp_callbacks(network_name):

	mlp_callbacks = [
		#EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=0, write_graph=False)	
	]
	return mlp_callbacks
