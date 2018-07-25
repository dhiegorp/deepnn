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
	'log_dir': base_path + '/logs/1layer/trigram/',
	'reports_dir': base_path + '/reports/1layer/trigram/',
	'fullds_reports_dir': base_path + '/reports/1layer/trigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/1layer/trigram/',
	'checkpoints_dir':base_path + '/checkpoints/1layer/trigram/',
	'executed_path':base_path + '/executed/1layer/trigram/',
	'data_dir': ds_path + '/',
	'fullds_data_dir': ds_path + '/',
	#'data_dir': ds_path + '/malware_selected_1gram_mini.pkl',
	#'fullds_data_dir':ds_path + '/malware_selected_1gram.pkl',
	

	'data_target_list' : [1,2,3,4,5,6,7,8,9],
	#'epochs': 50,
	'epochs': 200,
	#'epochs': 1000,
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
	#
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_1' : [10000, 1000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_2' : [10000, 2000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_3' : [10000, 3000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_4' : [10000, 4000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_5' : [10000, 5000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_6' : [10000, 6000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_7' : [10000, 7000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_8' : [10000, 8000],
	'AE_TRIGRAMA_1L_FULLDS_UNDER_F0_9' : [10000, 9000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_0' :  [10000,10000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_1' :  [10000,11000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_2' :  [10000,12000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_3' :  [10000,13000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_4' :  [10000,14000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_5' :  [10000,15000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_6' :  [10000,16000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_7' :  [10000,17000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_8' :  [10000,18000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F1_9' :  [10000,19000],
	'AE_TRIGRAMA_1L_FULLDS_OVER_F2_0' :  [10000,20000]

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

