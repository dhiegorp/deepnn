import numpy as np
from GLOBAL_EXP_FUNCTIONS import *
from TEST_GLOBAL import *
from trial import *
from datasets.dataset_loader import CSVDatasetLoader



def main():
	load_ds = CSVDatasetLoader(GLOBAL['data_dir'], 'malware_selected_1gram', resolve_names=True)
	trainx, trainy, valx, valy = load_ds()
	data = {'trainx': trainx, 'trainy': trainy, 'valx': valx, 'valy':valy}
	
	exp1 = Trial('teste_unigrama', [96,50,18,9], data=data, config_dict=GLOBAL)
	exp1()

if __name__ == '__main__':
	main()