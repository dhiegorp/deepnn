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
	'log_dir': base_path + '/logs/3layers/bigram/',
	'reports_dir': base_path + '/reports/3layers/bigram/',
	'fullds_reports_dir': base_path + '/reports/3layers/bigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/3layers/bigram/',
	'checkpoints_dir':base_path + '/checkpoints/3layers/bigram/',
	'executed_path':base_path + '/executed/3layers/bigram/',
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

	'AE_BIGRAMA_3L_MINIDS_OVER_01' : [9216, 9216, 8295, 7375],
	'AE_BIGRAMA_3L_MINIDS_OVER_02' : [9216, 14746, 13272, 11799],
	'AE_BIGRAMA_3L_MINIDS_OVER_03' : [9216, 15667, 14101, 12535],
	'AE_BIGRAMA_3L_MINIDS_OVER_04' : [9216, 16589, 14931, 13273],
	'AE_BIGRAMA_3L_MINIDS_OVER_05' : [9216, 18432, 16590, 14747],

	'AE_BIGRAMA_3L_FULLDS_OVER_01' : [9216, 9216, 8295, 7375],
	'AE_BIGRAMA_3L_FULLDS_OVER_02' : [9216, 14746, 13272, 11799],
	'AE_BIGRAMA_3L_FULLDS_OVER_03' : [9216, 15667, 14101, 12535],
	'AE_BIGRAMA_3L_FULLDS_OVER_04' : [9216, 16589, 14931, 13273],
	'AE_BIGRAMA_3L_FULLDS_OVER_05' : [9216, 18432, 16590, 14747]

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

