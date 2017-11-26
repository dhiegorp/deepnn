from dataset_loader import *
import math
import numpy as np
from sklearn.externals import joblib

def decode_onehot(vect, adjust=None):
	decoded = np.argmax(vect)
	if adjust:
		decoded = decoded + adjust
	return decoded

class CSVSampleSplitter:
	def __init__(self, path, dataset_filename_pattern, factor, train_factor, target_list, dump_to=None, file_pattern_dump=None, reshuffle=False):
			self.__path = path
			self.__dataset_filename_pattern = dataset_filename_pattern
			self.__factor = factor
			self.__train_factor = train_factor
			self.__reshuffle = reshuffle
			self.__target_list = target_list
			self.__dump_to= dump_to
			self._file_pattern_dump = file_pattern_dump
			self.__validate()

	def __validate(self):
		if self.__path == None:
			raise ValueError('the path for dataset was not configured!')
		
		if self.__dataset_filename_pattern == None:
			raise ValueError('the dataset filename pattern was not configured!')

		if self.__factor == None:
			raise ValueError('a factor is required for slicing the dataset')

		if self.__target_list == None:
			raise ValueError('a target list must be provided')

	def __load(self):

		ds_loader = CSVDatasetLoader(self.__path, self.__dataset_filename_pattern, resolve_names=True)
		self.__Xt, self.__yt, self.__Xv, self.__yv = ds_loader()
		msg = """
		=======================================
		loading malware dataset on = {}
		filename pattern = {}	
		trainx shape = {}
		trainy shape = {}
		valx shape = {}
		valy shape = {}
		=======================================
			""".format(self.__path, self.__dataset_filename_pattern, str(self.__Xt.shape), str(self.__yt.shape), str(self.__Xv.shape), str(self.__yv.shape))
		print(msg)


	def __gen_counter_obj(self):
		cnt = {}

		for c in self.__target_list:
			cnt[c] = 0

		return cnt

	def __count(self):
		
		cnt = self.__gen_counter_obj()
		print('cnt >>> ', cnt)
		for i in self.__yt:
			for c,v in cnt.items():
				
				if c == decode_onehot(i, adjust=1):
					cnt[c] = v + 1

		for i in self.__yv:
			for c,v in cnt.items():
				if c == decode_onehot(i, adjust=1):
					cnt[c] = v + 1		

		cc = {}	

		for k,v in cnt.items():

			final_num = math.ceil(v * self.__factor)
			train_factor = math.ceil(final_num * self.__train_factor)
			validation_samples = math.fabs(final_num - train_factor)
			print(':::: for class ', k, 'total sliced:', final_num, ', training:', train_factor, ' - validation:', validation_samples)

			cc[k] = {'total': v, 'train': train_factor, 'validation': math.fabs(final_num - train_factor) }

		print('returning ', cc)
		return cc


	def __get_samples(self, counter):
		xval = []
		yval = []
		xtra = []
		ytra = []


		for k,v in counter.items():
			print('training :: getting ', v['train'], ' samples for class ', k, ' self.__Xt : ', self.__Xt.shape)
			acc = 1
			for num, row in enumerate(self.__Xt):
				if decode_onehot(self.__yt[num], adjust=1) == k and acc <= v['train']:
					xtra.append(row)
					ytra.append(self.__yt[num])
					acc = acc + 1

			print('training :: ending for class ', k, ' on training! (', v['train'], ' -xtra: ', len(xtra), ' acc ', acc , ' )')

			print('validation :: getting ', v['validation'], ' samples for class ', k, ' self.__Xv : ', self.__Xv.shape)
			acc = 1
			for num, row in enumerate(self.__Xv):
				if decode_onehot(self.__yv[num], adjust=1) == k and acc <= v['validation']:
					xval.append(row)
					yval.append(self.__yv[num])
					acc = acc + 1
			print('validation :: ending for class ', k, ' on validation! (', v['validation'], ' -xval: ', len(xval), ' acc ', acc , ' )')

			

		self.__xtfinal = np.array(xtra)
		self.__ytfinal = np.array(ytra)
		self.__xvfinal = np.array(xval)
		self.__yvfinal = np.array(yval)

		print('xtfinal> ', self.__xtfinal.shape)
		print('ytfinal> ', self.__ytfinal.shape)
		print('xvfinal> ', self.__xvfinal.shape)
		print('yvfinal> ', self.__yvfinal.shape)

		#ret = ( np.c_[self.__xtfinal, self.__ytfinal] , np.c_[self.__xvfinal, self.__yvfinal])
		#joblib.dump(ret, self.__dump_to)
		np.savetxt(self.__dump_to + self._file_pattern_dump + '_train_feat.csv', self.__xtfinal, delimiter=',')
		np.savetxt(self.__dump_to + self._file_pattern_dump + '_train_target.csv', self.__ytfinal, delimiter=',')
		np.savetxt(self.__dump_to + self._file_pattern_dump + '_test_feat.csv', self.__xvfinal, delimiter=',')
		np.savetxt(self.__dump_to + self._file_pattern_dump + '_test_target.csv', self.__yvfinal, delimiter=',')
 


	def process(self):
		self.__load()
		count = self.__count()
		print(count)		
		self.__get_samples(count)




class SampleSplitter:
	def __init__(self, dataset, factor, train_factor, target_list, dump_to=None, reshuffle=False):
			self.__dataset = dataset
			self.__factor = factor
			self.__train_factor = train_factor
			self.__reshuffle = reshuffle
			self.__target_list = target_list
			self.__dump_to= dump_to
			self.__validate()

	def __validate(self):
		if self.__dataset == None:
			raise ValueError('the path for dataset was not configured!')

		if self.__factor == None:
			raise ValueError('a factor is required for slicing the dataset')

		if self.__target_list == None:
			raise ValueError('a target list must be provided')

	def __load(self):

		ds_loader = DatasetLoader(self.__dataset, targets_list=self.__target_list, normalize=True, maintain_originals=True, one_hot_encoding=False)
		self.__Xt, self.__yt, self.__Xv, self.__yv = ds_loader()
		msg = """
	=======================================
	loading malware dataset on = {}	
	trainx shape = {}
	trainy shape = {}
	valx shape = {}
	valy shape = {}
	=======================================
			""".format(self.__dataset, str(self.__Xt.shape), str(self.__yt.shape), str(self.__Xv.shape), str(self.__yv.shape))
		print(msg)

	def __gen_counter_obj(self):
		cnt = {}

		for c in self.__target_list:
			cnt[c] = 0

		return cnt

	def __count(self):
		
		cnt = self.__gen_counter_obj()

		for i in self.__yt:
			for c,v in cnt.items():
				if c == i:
					cnt[c] = v + 1

		for i in self.__yv:
			for c,v in cnt.items():
				if c == i:
					cnt[c] = v + 1		

		cc = {}	

		

		for k,v in cnt.items():
			final_num = math.ceil(v * self.__factor)
			train_factor = math.ceil(final_num * self.__train_factor)
			cc[k] = {'total': v, 'train': train_factor, 'validation': math.fabs(final_num - train_factor) }

		return cc

	def __get_samples(self, counter):
		xval = []
		yval = []
		xtra = []
		ytra = []

		for k,v in counter.items():
			print('getting ', v['train'], ' samples for class ', k)
			acc = 1
			for num, row in enumerate(self.__Xt):
				if self.__yt[num] == k and acc < v['train']:
					xtra.append(row)
					ytra.append(self.__yt[num])
					acc = acc + 1

			acc = 1
			for num, row in enumerate(self.__Xv):
				if self.__yv[num] == k and acc < v['validation']:
					xval.append(row)
					yval.append(self.__yv[num])
					acc = acc + 1
			

		self.__xtfinal = np.array(xtra)
		self.__ytfinal = np.array(ytra)
		self.__xvfinal = np.array(xval)
		self.__yvfinal = np.array(yval)

		print('xtfinal> ', self.__xtfinal.shape)
		print('ytfinal> ', self.__ytfinal.shape)
		print('xvfinal> ', self.__xvfinal.shape)
		print('yvfinal> ', self.__yvfinal.shape)

		ret = ( np.c_[self.__xtfinal, self.__ytfinal] , np.c_[self.__xvfinal, self.__yvfinal])
		
		joblib.dump(ret, self.__dump_to)
 


	def process(self):
		self.__load()
		count = self.__count()
		print(count)		
		self.__get_samples(count)



def main():

	print('Oh my new version!')

	process_list = [
		#('e:/research/malware_dataset/malware_selected_1gram.pkl', 'e:/research/malware_dataset/malware_selected_1gram_mini.pkl'),
		('/home/dhiegorp/malware_dataset/malware_selected_2gram.pkl', 'c:/Users/dhieg/research/malware_dataset/malware_selected_2gram_mini_2.pkl'),
		#('e:/research/malware_dataset/malware_selected_3gram.pkl', 'e:/research/malware_dataset/malware_selected_3gram_mini.pkl')
	]	

	TOTAL_SLICE = 0.15
	TRAIN_SPLIT = 0.75
	CLASS_LIST = [1,2,3,4,5,6,7,8,9]
	RESHUFFLE = False

	print('loading data...')
	ss = CSVSampleSplitter('/home/dhiegorp/malware_dataset/', 
		'malware_selected_2gram',  
		factor=TOTAL_SLICE, 
		train_factor=TRAIN_SPLIT, 
		target_list=CLASS_LIST,
		file_pattern_dump='malware_selected_2gram_mini2',
		dump_to='/home/dhiegorp/malware_dataset/')
	print('starting process...')
	ss.process()
	print('done!')

if __name__ == '__main__':
	main()