import os.path
import sys
import logging
from GLOBAL_EXP_FUNCTIONS import *
import numpy as np
from deepnn.autoencoders.Autoencoder import Autoencoder
from datasets.dataset_loader import CSVDatasetLoader
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

np.random.seed(666)



"""
	Trial('unigrama_01', [96,50,10,25], )
"""
class Trial:
	def __init__(self, trial_name, topology, data={}, config_dict=None):
		self.__config = config_dict
		self.__data = data
		self.__name = trial_name
		self.__topology = topology

		self.__setup_log()
		


	def __network_name(self):
		return self.__name 

	def __network_name_path(self):
		return self.__config['executed_path'] + self.__network_name()


	def __setup_log(self):
		logging.basicConfig(format=self.__config['log_format'], filename=self.__config['log_dir'] + self.__name + '.log', level=logging.DEBUG)	

	def __header_log(self):
		header = """
		=======================================
		network_name = {}
		layers = {}
		using GLOBAL obj = 
			{}
		=======================================
		"""
		print(self.__topology)
		print(self.__config)
		logging.debug(header.format(self.__name, ','.join(str(layer) for layer in self.__topology),  str(self.__config)))
	
	def __train_and_eval_autoencoder(self):
		 

		logging.debug("=======================================")
		
		CONFIG = self.__config['autoencoder_configs']
	
		logging.debug("setting configurations for autoencoder: \n\t " + str(CONFIG) )
	
		self.__ae_model = Autoencoder(
			self.__topology, 
			name = self.__name, 
			hidden_layer_activation = CONFIG['hidden_layer_activation'], 
			output_layer_activation = CONFIG['output_layer_activation'], 
			loss_function = CONFIG['loss_function'], 
			optimizer = CONFIG['optimizer'], 
			discard_decoder_model = CONFIG['discard_decoder_function'])

		logging.debug("training and evaluating autoencoder")	

		self.__ae_model.train_and_eval(
			feature=self.__data['trainx'], 
			feature_validation=self.__data['valx'], 
			epochs=self.__config['epochs'], 
			batch_size=self.__config['batch'], 
			shuffle=self.__config['shuffle_batches'],   
			store_history=self.__config['store_history'], 
			callbacks = get_ae_callbacks(self.__name, self.__config['checkpoints_dir'], self.__config['tensorflow_dir']) )

		logging.debug("trained and evaluated!")	
		logging.debug("MODEL FOR AE {}, TOPOLOGY:\n\t\t{}".format(self.__network_name, self.__ae_model.summary)

		try: 
			logging.debug("Training history: \n" + str(self.__ae_model.training_history.history) )
		except:
			pass

		logging.debug("done!")

	def __train_and_eval_mlp(self):


		logging.debug("=======================================")

		CONFIG = self.__config['mlp_configs']

		logging.debug("setting configurations for classifier: \n\t " + str(self.__config) )

		self.__mlp_model = self.__ae_model.get_classifier( 
			activation = CONFIG['activation'], 
			loss_function = CONFIG['loss_function'], 
			optimizer = CONFIG['optimizer'], 
			use_last_dim_as_classifier_dim = CONFIG['use_last_dim_as_classifier'], 
			classifier_dim = CONFIG['classifier_dim'])

		logging.debug("training ... ")	

		self.__mlp_model.train( 
			feature=self.__data['trainx'], 
			label=self.__data['trainy'],   
			validation=(self.__data['valx'], self.__data['valy']), 
			epochs=self.__config['epochs'], 
			batch_size=self.__config['batch'], 
			shuffle=self.__config['shuffle_batches'], 
			store_history=self.__config['store_history'],
			callbacks=get_mlp_callbacks(self.__name, self.__config['checkpoints_dir'], self.__config['tensorflow_dir']) )

		logging.debug("trained!")

		logging.debug("MODEL FOR MLP {}, TOPOLOGY:\n\t\t{}".format(self.__network_name, self.__mlp_model.summary)
			
		
		try: 
			logging.debug("Training history: \n" + str(self.__ae_model.training_history.history) )
		except:
			pass

		logging.debug('evaluating model ... ')
		
		self.__mlp_model.eval(feature=self.__data['valx'], label=self.__data['valy'] )

		logging.debug('evaluated! ')

		logging.debug('generating reports ... ')
		self.__mlp_model.eval_stats(self.__config['reports_dir'])

		logging.debug('done!')
			

	def __execute_trial(self):
		if is_executed(self.__network_name_path()):
			logging.debug("The experiment " + self.__name + " was already executed!")
		else:
			logging.debug(">> Initializing execution of experiment " + self.__name )
			logging.debug(">> Printing header log")
			self.__header_log()
			#logging.debug(">> Loading dataset... ")
			#data_init()

			logging.debug(">> Executing autoencoder part ... ")
			self.__train_and_eval_autoencoder()
			logging.debug(">> Executing classifier part ... ")
			self.__train_and_eval_mlp()
			logging.debug(">> experiment " + self.__network_name() + " finished!")
			mark_as_done(self.__network_name_path())


	@property
	def name(self):
		return self.__name

	def __call__(self):
		self.__execute_trial()


