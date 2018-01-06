import glob
import numpy as np

CONFUSION_MATRIX_LATEX_FMT = """
\\begin{{table}}[ht]
\\centering
\\caption{{{}}}
\\label{{{}}}
\\begin{{tabular}}{{c|c|c|c|c|c|c|c|c|c|}}
\\cline{{2-10}}
\\multicolumn{{1}}{{l|}}{{}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{1}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{2}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{3}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{4}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{5}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{6}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{7}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{8}} & \\cellcolor[HTML]{{EFEFEF}}\\textbf{{9}} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{1}}}} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{2}}}} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{3}}}} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{4}}}} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{5}}}} & {} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{6}}}} & {} & {} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{7}}}} & {} & {} & {} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{8}}}} & {} & {} & {} & {} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} & {} \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{9}}}} & {} & {} & {} & {} & {} & {} & {} & {} & \\cellcolor[HTML]{{333333}}{{\\color[HTML]{{FFFFFF}} {}}} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""

TABLE_LABEL_FMT = "conf:{}"
TABLE_CAPTION_FMT = "Resultado do experimento {}, em matriz de confusão, na fase de validação do modelo."



def conf_matrix_parser(path):
	matrix = []
	with open(path, 'r') as file:
		for id, line in enumerate(file):
			if id > 10:
				raise Exception("weird pattern found")
			if id >= 2:
				matrix.append([item.rstrip() for i,item in enumerate(line.split(' ')) if i != 0 and item != '']) 
	return matrix

def confusion_matrix_latex_fmt(path):
	with open(path, 'r') as file:
		data = "".join(content.rstrip() for content in file)
		return data

def create_confusion_matrix(matrix, experiment_name):
	cf = tuple(np.array(matrix).reshape(-1,))
	xp = experiment_name_for_latex(experiment_name)
	caption = TABLE_CAPTION_FMT.format(xp)
	label = TABLE_LABEL_FMT.format(xp)

	return (caption, label) + cf

def experiment_name_for_latex(name):
	return name.replace('_','\_')

def seek_and_destroy(initial_path):
	found = glob.iglob(initial_path + '*.confusion_matrix.txt')
	print('Files found that match the criteria:\n\t' + ',\n\t'.join( n for n in found ) )
	
	print('teste')
	found = glob.iglob(initial_path + '*.confusion_matrix.txt')
	for name in found:
			
		experiment_name = name.split()[0]
		
		xpname_latex = name.split('\\')
		xpname_latex = xpname_latex[len(xpname_latex)-1].split('.')[0]

		print(xpname_latex, ' selected! Recovering confusion matrix values')
		matrix = conf_matrix_parser(name)
		print('matrix recovered: ', matrix)

		with open(experiment_name+'.cf.txt', 'a') as file:
			print('oopsy doopsy')
			new_cf = CONFUSION_MATRIX_LATEX_FMT.format(*create_confusion_matrix(matrix, xpname_latex))
			file.write(new_cf)

def test_conf_matrix_parser(name):
	print('testing name conf_matrix_parser with name ', name)
	x = conf_matrix_parser(name)
	print(x)

if __name__ == '__main__':
	
	file_fullpath = 'C:/Users/dhieg/research/research_msc/reports/1layer/unigram/fullds/AE_UNIGRAMA_1L_OVER_F1_0.confusion_matrix.txt'
	seek_and_destroy('C:/Users/dhieg/research/research_msc/reports/1layer/unigram/fullds/')

