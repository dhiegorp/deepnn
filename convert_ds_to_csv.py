import sys 
import numpy as np
from datasets.dataset_loader import *
 
def main():
	pkl_path = sys.argv[1]
	csv_path = pkl_path.split('.')[0]

	msg = """
		Converting pkl file (numpy matrix) to CSV
		file path: {} 

		""".format(pkl_path)

	print(msg)

	ds = DatasetLoader(pkl_path, targets_list=[1,2,3,4,5,6,7,8,9], normalize=True, maintain_originals=True)
	tx, ty, vx, vy = ds()

	msg = """
			The dataset was loaded!
			Train features dim {}
			Train target {}
			Validation features dim {}
			Validation target {}
		""".format(tx.shape,ty.shape,vx.shape,vy.shape)
	print(msg)
	print('creating csvs... ')
	np.savetxt(csv_path + '_train_feat.csv', tx, delimiter=',')
	np.savetxt(csv_path + '_train_target.csv', ty, delimiter=',')
	np.savetxt(csv_path + '_test_feat.csv', vx, delimiter=',')
	np.savetxt(csv_path + '_test_target.csv', vy, delimiter=',')
	print('done!')
if __name__ == '__main__':
	main()