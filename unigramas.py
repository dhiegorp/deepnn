from ENVIRONMENT import *

from trial import *



def load_datasource(source, config):
	global data

	load_ds = CSVDatasetLoader(config['data_dir'], source, resolve_names=True)
	trainx, trainy, valx, valy = load_ds()
	data = {'trainx': trainx, 'trainy': trainy, 'valx': valx, 'valy':valy}


def configure_trials(name_pattern, data, config, hidden_layers):
	trials =[]
	for i, hidden_layer in enumerate(hidden_layers):
		print('ae_bigrama_1L_', i, ' - hidden:', hidden_layer[0])
		trials.append( Trial('ae_bigrama_1L_' + i, [96].extend(hidden_layer[0]), data=data, config_dict=config) )
	return trials


def execute_trials(trials):
	for i, trial_exec in enumerate(trials):
		print('executing trial ', i, ', named: ', trial_exec.name)
		trial_exec()


def nn_1l():
	from GLOBAL_EXP_CONFIG_1L_UNIGRAM import GLOBAL as GL1
	
	hidden_configs = [ 
	[9],[19],[28],[38],
	[48],[57],[67],[76],
	[86],[96],[105],[115],
	[124],[134],[144],[153],
	[163],[172],[182],[192] ]

	load_datasource('malware_selected_1gram', GL1)

	trials = configure_trials('ae_unigrama_1L_', data, GL1, hidden_configs)

	if not trials:
		print('No trials configured for hidden_configs ', str(hidden_configs))
	else:
		execute_trials(trials)



def main():
	nn_1l()	


if __name__ == '__main__':
	main()

"""
	exp1()
exp1 = Trial('teste_unigrama', [96,50,18,9], data=data, config_dict=GLOBAL)
"""
