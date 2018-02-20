from pathlib import Path

str_pattern = "{}\t\t[{}]\t"

function verify(imp_global):
	
	for x in imp_global.MAP_DIMS:
		path_to_file = b1.executed_path + x[0]
		exists = 'NAAAHHH!'
		if Path(path_to_file).exists():
			exists = 'EXECUTED' 

		print(str_pattern.format(x[0], exists))

function rastrear_execucoes_bigramas():
	import GLOBAL_1L_BIGRAM as b1
	import GLOBAL_2L_BIGRAM as b2
	import GLOBAL_3L_BIGRAM as b3
	import GLOBAL_4L_BIGRAM as b4
	import GLOBAL_5L_BIGRAM as b5
	import GLOBAL_6L_BIGRAM as b6
	import GLOBAL_7L_BIGRAM as b7
	import GLOBAL_8L_BIGRAM as b8
	import GLOBAL_9L_BIGRAM as b9
	import GLOBAL_10L_BIGRAM as b10

	print('BIGRAM NETS')
	print('==============================\n')
	verify(b1)
	verify(b2)
	verify(b3)
	verify(b4)
	verify(b5)
	verify(b6)
	verify(b7)
	verify(b8)
	verify(b9)
	verify(b10)
