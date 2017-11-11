from ENVIRONMENT import *
from GLOBAL_EXP_CONFIG_1L_BIGRAM import *
from trial import *


def main():
	
	#hidden_configs = [
	#	922, 1843, 2765, 
	#	3686, 4608, 5530, 
	#	6451, 7373, 8294, 
	#	9216, 10138, 11059, 
	#	11981, 12902, 13824, 
	#	14746, 15667, 16589, 
	#	17510, 18432]
	
	hidden_configs = [
		1,2,3]
	

	trials = []

	load_ds = CSVDatasetLoader(GLOBAL['data_dir'], 'malware_selected_2gram_mini', resolve_names=True)
	trainx, trainy, valx, valy = load_ds()
	data = {'trainx': trainx, 'trainy': trainy, 'valx': valx, 'valy':valy}
	

	for i, item in hidden_configs:
		print('ae_bigrama_1L_', i, ' - hidden:', item)
		trials.append( Trial('ae_bigrama_1L_' + i, [96, item], data=data, config_dict=GLOBAL) )

	for i, item in trials:
		trials[i]()


if __name__ == '__main__':
	main()

"""
	exp1()
exp1 = Trial('teste_unigrama', [96,50,18,9], data=data, config_dict=GLOBAL)
"""
