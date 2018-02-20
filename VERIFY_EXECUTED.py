from pathlib import Path

str_pattern = "\t{}\t\t[{}]\t"

def verify(imp_global):
	
	for x in imp_global.MAP_DIMS:
		path_to_file = b1.executed_path + x[0]
		exists = 'NAAAHHH!'
		if Path(path_to_file).exists():
			exists = 'EXECUTED' 

		print(str_pattern.format(x[0], exists))

def rastrear_execucoes_bigramas():
	import GLOBAL_EXP_CONFIG_1L_BIGRAM as b1
	import GLOBAL_EXP_CONFIG_2L_BIGRAM as b2
	import GLOBAL_EXP_CONFIG_3L_BIGRAM as b3
	import GLOBAL_EXP_CONFIG_4L_BIGRAM as b4
	import GLOBAL_EXP_CONFIG_5L_BIGRAM as b5
	import GLOBAL_EXP_CONFIG_6L_BIGRAM as b6
	import GLOBAL_EXP_CONFIG_7L_BIGRAM as b7
	import GLOBAL_EXP_CONFIG_8L_BIGRAM as b8
	import GLOBAL_EXP_CONFIG_9L_BIGRAM as b9
	import GLOBAL_EXP_CONFIG_10L_BIGRAM as b10

	print('BIGRAM NETS')
	print('==============================\n')
	print('1L\n')
	verify(b1)
	print('2L\n')
	verify(b2)
	print('3L\n')
	verify(b3)
	print('4L\n')
	verify(b4)
	print('5L\n')
	verify(b5)
	print('6L\n')
	verify(b6)
	print('7L\n')
	verify(b7)
	print('8L\n')
	verify(b8)
	print('9L\n')
	verify(b9)
	print('10L\n')
	verify(b10)


def main():
	rastrear_execucoes_bigramas()

if __name__ == '__main__':
	main()