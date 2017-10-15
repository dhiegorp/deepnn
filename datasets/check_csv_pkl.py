import sys 
import numpy as np
from dataset_loader import *
import glob

def main():
	targets=[1,2,3,4,5,6,7,8,9]

	print('\n\nChecking if converted csvs are equal to pickle stored datasets\n\n')

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_1gram.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_1gram', resolve_names=True)()
	print('1gram     : pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_1gram_mini.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_1gram_mini', resolve_names=True)()
	print('1gram mini: pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_2gram.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_2gram', resolve_names=True)()
	print('2gram     : pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_2gram_mini.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_2gram_mini', resolve_names=True)()
	print('2gram mini: pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_3gram.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_3gram', resolve_names=True)()
	print('3gram     : pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))

	tx, ty, vx, vy = DatasetLoader('data/malware_selected_3gram_mini.pkl', targets_list=targets, normalize=True, maintain_originals=True)()
	ttx, tty, vvx, vvy = CSVDatasetLoader('data/', 'malware_selected_3gram_mini', resolve_names=True)()
	print('3gram mini: pkl == csvs [{}]'.format( (tx==ttx).all() and (ty==tty).all() and (vx==vvx).all() and (vy==vvy).all() ))


if __name__ == '__main__':
	main()