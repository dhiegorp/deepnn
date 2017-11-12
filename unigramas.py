from ENVIRONMENT import *

from trial import *



def load_datasource(source, config):
	global data

	load_ds = CSVDatasetLoader(config['data_dir'], source, resolve_names=True)
	trainx, trainy, valx, valy = load_ds()
	data = {'trainx': trainx, 'trainy': trainy, 'valx': valx, 'valy':valy}


def configure_trials(name_pattern, input_dim,  data, config, hidden_layers):
	trials =[]
	for i, hidden_layer in enumerate(hidden_layers):
		print('ae_bigrama_1L_',i, ' - hidden:', hidden_layer)
		topol = [input_dim]
		topol.extend(hidden_layer) 
		trials.append( Trial('ae_unigrama_1L_' + str(i), topol, data=data, config_dict=config) )
	return trials

def alt_execute_trials(name_pattern, input_dim,  data, config, hidden_layers):
	for i, hidden_layer in enumerate(hidden_layers):
		print('ae_unigrama_1L_',i, ' - hidden:', hidden_layer)
		topol = [input_dim]
		topol.extend(hidden_layer) 
		experiment = Trial('ae_unigrama_1L_' + str(i), topol, data=data, config_dict=config)
		experiment()
		del experiment
		del topol



def execute_trials(trials):
	for i, trial_exec in enumerate(trials):
		print('executing trial ', i, ', named: ', trial_exec.name)
		#try:
		trial_exec()
		#except:
		#		print('error executing')
		#		pass


def nn_1l():
	from GLOBAL_EXP_CONFIG_1L_UNIGRAM import GLOBAL as GL1
	
	hidden_configs = [ 
	[9],[19],[28],[38],
	[48],[57],[67],[76],
	[86],[96],[105],[115],
	[124],[134],[144],[153],
	[163],[172],[182],[192] ]

	print('nn1l hidden configs: ', hidden_configs)

	load_datasource('malware_selected_1gram', GL1)
	#alt_execute_trials('ae_unigrama_1L_', 96, data, GL1, hidden_configs)

	Trial('ae_unigrama_1L_' + str(18), [96,182], data=data, config_dict=GL1)()
	Trial('ae_unigrama_1L_' + str(19), [96,192], data=data, config_dict=GL1)()

def main():
	nn_1l()	


if __name__ == '__main__':
	main()

"""
	exp1()
exp1 = Trial('teste_unigrama', [96,50,18,9], data=data, config_dict=GLOBAL)
"""
