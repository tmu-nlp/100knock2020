file = open('popular-names.txt', 'r')
lines = [line for line in file.readlines()]

file_col1 = open('col1.txt', 'w')
file_col2 = open('col2.txt', 'w')

for line in lines:
    line = line.split()
    file_col1.write(line[0] + '\n')
    file_col2.write(line[1] + '\n')

#cut -f (range) -d (delimiter)
#cut -f 1 -d " " popular-names.txt > col1.txt
#cut -f 2 -d " " popular-names.txt > col2.txt