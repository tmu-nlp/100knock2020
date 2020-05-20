import sys

file_names = open(sys.argv[1], 'r')
name_lines = [line.replace('\n', '') for line in file_names.readlines()]

n = int(input())

names_n_divided = []
for i in range(n):
    names_n_divided.append(name_lines[i::n])

print(names_n_divided)

#wc -l merge.txt
#split -l [] merge.txt [prefix]
#-n optionはなくなったようだ...