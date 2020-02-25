def count_char(fn):
	import os.path
	if os.path.isfile(fn):
		with open(fn,'r') as fh:
			total = 0
			for line in fh:
				for char in ('#','/','\n',':'):
					line = line.replace(char," ")
				total += len(line.strip().split())
			return total 
print(count_char('./readme.md'))

