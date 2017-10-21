import os.path
import sys
import logging
from GLOBAL_EXP_FUNCTIONS import *
import GLOBAL_EXP_CONFIG_2L_UNIGRAM as g2l
import GLOBAL_EXP_CONFIG_3L_UNIGRAM as g3l
import GLOBAL_EXP_CONFIG_4L_UNIGRAM as g4l
import GLOBAL_EXP_CONFIG_5L_UNIGRAM as g5l
import GLOBAL_EXP_CONFIG_6L_UNIGRAM as g6l
import GLOBAL_EXP_CONFIG_7L_UNIGRAM as g7l
import GLOBAL_EXP_CONFIG_8L_UNIGRAM as g8l
import GLOBAL_EXP_CONFIG_9L_UNIGRAM as g9l
import GLOBAL_EXP_CONFIG_10L_UNIGRAM as g10l
import numpy as np
from deepnn.autoencoders.Autoencoder import Autoencoder
from datasets.dataset_loader import CSVDatasetLoader
import keras
from keras.models import Model
from keras.layers import Input, Dense
from pandas_ml import ConfusionMatrix
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import glob
import time


minids_trainx, minids_trainy, minids_valx, minids_valy, minids_load_ds = None, None, None, None, None
fullds_trainx, fullds_trainy, fullds_valx, fullds_valy, fullds_load_ds = None, None, None, None, None

def data_init(GLOBAL):
	global minids_trainx, minids_trainy, minids_valx, minids_valy, minids_load_ds, fullds_trainx, fullds_trainy, fullds_valx, fullds_valy, fullds_load_ds
	minids_load_ds = CSVDatasetLoader(GLOBAL['data_dir'], 'malware_selected_1gram_mini', resolve_names=True)
	fullds_load_ds = CSVDatasetLoader(GLOBAL['fullds_data_dir'], 'malware_selected_1gram', resolve_names=True)
	
	#minids_load_ds = DatasetLoader(GLOBAL['data_dir'], targets_list=GLOBAL['data_target_list'], normalize=True, maintain_originals=True)
	#fullds_load_ds = DatasetLoader(GLOBAL['fullds_data_dir'], targets_list=GLOBAL['data_target_list'], normalize=True, maintain_originals=True)
	minids_trainx, minids_trainy, minids_valx, minids_valy = minids_load_ds()
	fullds_trainx, fullds_trainy, fullds_valx, fullds_valy = fullds_load_ds()
	msg = """
	datasource: {}
	=======================================
	loading malware dataset on = {}	
	trainx shape = {}
	trainy shape = {}
	valx shape = {}
	valy shape = {}
	=======================================
	"""

	print(msg.format('completo ', GLOBAL['fullds_data_dir'], str(fullds_trainx.shape), str(fullds_trainy.shape), str(fullds_valx.shape), str(fullds_valy.shape)))
	print(msg.format('mini ', GLOBAL['data_dir'], str(minids_trainx.shape), str(minids_trainy.shape), str(minids_valx.shape), str(minids_valy.shape)))


def build_mlp_model(dims, GLOBAL):

	input = Input(shape=(dims[0],))
	encoder_layers = None

	for id, neurons in enumerate(dims[1:]):
		if id == 0:
			encoder_layers = Dense(neurons, activation='relu', name='enc{}_{}'.format(id, neurons) )(input)
		else:
			encoder_layers = Dense(neurons, activation='relu', name='enc{}_{}'.format(id, neurons) )(encoder_layers)

	encoder_layers = Dense(GLOBAL['mlp_configs']['classifier_dim'], activation=GLOBAL['mlp_configs']['activation'], name='classifier')(encoder_layers)

	
	model = Model(inputs=[input], outputs=[encoder_layers])

	return model

def load_checkpoint(name, MAP_DIMS, GLOBAL):
	model = build_mlp_model(MAP_DIMS[name], GLOBAL)	
	model.load_weights(get_checkpoint_path(name, GLOBAL))
	model.compile(loss=GLOBAL['mlp_configs']['loss_function'], optimizer=GLOBAL['mlp_configs']['optimizer'], metrics=['acc'])
	return model


def get_checkpoint_path(name, GLOBAL):
	return GLOBAL['checkpoints_dir'] + name.split('.')[0] + '_mlp.h5' 

def stats( name, predictions, label, path):
			
	file_pattern = path + name + '.{0}.{1}'

	classifier_predictions_max = np.argmax(predictions, axis=1)
	Ymax = np.argmax(label, axis=1)

	confusion_matrix = ConfusionMatrix(Ymax, classifier_predictions_max)

	status_dump(file_pattern, confusion_matrix, html=True, string=True, pickle=True, stats_as_txt=True, latex=True)

	


def status_dump( file_pattern, confusion_matrix , html=False,string=False, pickle=False, stats_as_txt=False, latex=False):
	dataframe = confusion_matrix.to_dataframe()
	
	if html:
		with open(file_pattern.format('confusion_matrix','html'),'w') as file:
			file.write(dataframe.to_html())
	if string:
			with open(file_pattern.format('confusion_matrix','txt'),'w') as file:
				file.write(dataframe.to_string())
	if pickle:
			dataframe.to_pickle(file_pattern.format('confusion_matrix','pickle'))

	if stats_as_txt:
		with open(file_pattern.format('stats','.txt'),'w') as file:
				file.write(str(confusion_matrix.stats()))
	if latex:
		with open(file_pattern.format('confusion_matrix','.latex_table'),'w') as file:
				file.write(dataframe.to_latex())


def evaluate(name, MAP_DIMS, GLOBAL):

	model = load_checkpoint(name, MAP_DIMS, GLOBAL)
	predict_full = model.predict(fullds_valx)
	predict_mini = model.predict(minids_valx)
	stats(name, predict_full, fullds_valy, GLOBAL['fullds_reports_dir'])
	#stats(name, predict_mini, minids_valy, GLOBAL['reports_dir'])

def nn_layer(layer, MAP_DIMS, GLOBAL):
	start = time.time()
	files = glob.glob('AE_UNIGRAMA_{}L*.py'.format(layer))
	print('selected files : ', files)
	data_init(GLOBAL)
	for f in files:
		evaluate(f.split('.')[0], MAP_DIMS, GLOBAL)
	end = time.time()-start
	print('the evaluation for fullds and minids for ', len(files), ' model(s) took ', end, ' second(s)')
	print('model list: \n\t\t', files)

def execute():

	nn_layer(2, g2l.MAP_DIMS, g2l.GLOBAL)
	nn_layer(3, g3l.MAP_DIMS, g3l.GLOBAL)
	nn_layer(4, g4l.MAP_DIMS, g4l.GLOBAL)
	nn_layer(5, g5l.MAP_DIMS, g5l.GLOBAL)
	nn_layer(6, g6l.MAP_DIMS, g6l.GLOBAL)
	nn_layer(7, g7l.MAP_DIMS, g7l.GLOBAL)
	nn_layer(8, g8l.MAP_DIMS, g8l.GLOBAL)
	nn_layer(9, g9l.MAP_DIMS, g9l.GLOBAL)
	nn_layer(10, g10l.MAP_DIMS, g10l.GLOBAL)

if __name__ == '__main__':
	execute()