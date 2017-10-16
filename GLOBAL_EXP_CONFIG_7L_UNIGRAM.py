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
	'log_dir': base_path + '/logs/7layers/unigram/',
	'reports_dir': base_path + '/reports/7layers/unigram/',
	'fullds_reports_dir': base_path + '/reports/7layers/unigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/7layers/unigram/',
	'checkpoints_dir':base_path + '/checkpoints/7layers/unigram/',
	'executed_path':base_path + '/executed/7layers/unigram/',
	'data_dir': ds_path + '/',
	'fullds_data_dir': ds_path + '/',
	#'data_dir': ds_path + '/malware_selected_1gram_mini.pkl',
	#'fullds_data_dir':ds_path + '/malware_selected_1gram.pkl',
	

	'data_target_list' : [1,2,3,4,5,6,7,8,9],
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
	'AE_UNIGRAMA_7L_UNDER_96_76_66_56_46_36_26_16_9' :  [96,  76,  66, 56, 46, 36, 26, 16, 9],
	'AE_UNIGRAMA_7L_UNDER_96_28_25_22_19_16_13_10_9' :  [96,  28,  25, 22, 19, 16, 13, 10, 9],
	'AE_UNIGRAMA_7L_OVER_96_134_124_114_104_94_84_74_9': [96, 134, 124, 114, 104, 94, 84, 74, 9],
	'AE_UNIGRAMA_7L_OVER_96_172_162_152_142_132_122_112_9': [96, 172, 162, 152, 142, 132, 122, 112, 9],
	'AE_UNIGRAMA_7L_UNDER_96_86_76_66_56_46_36_26_9' :  [96,  86,  76, 66, 56, 46, 36, 26, 9]
}

def get_ae_callbacks(network_name):
	ae_callbacks = [
		EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=1, write_graph=True)	
	]
	return ae_callbacks

def get_mlp_callbacks(network_name):

	mlp_callbacks = [
		EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=1, write_graph=True)	
	]
	return mlp_callbacks

