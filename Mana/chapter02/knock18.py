file = open('popular-names.txt', 'r')
lines = [line.replace('\n', '').split(' ') for line in file.readlines()]

dict_line = {}

for line in lines:
    dict_line[line[2]] = line

file_sorted = open('popular-names_sorted.txt', 'w')
for elem in sorted(dict_line):
    file_sorted.write('\t'.join(dict_line[elem]) + '\n')

#-k [] : sort by the value of []th field, n : as an int, r : reversed = True
#sort -k 3nr popular-names.txt