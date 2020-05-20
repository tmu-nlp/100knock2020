import sys

file = open(sys.argv[1], 'r')
col1 = []
col2 = []

for line in file:
    cols = line.split('\t')
    col1.append(cols[0])
    col2.append(cols[1])

col1_file = open('col1.txt', 'w')
col2_file = open('col2.txt', 'w')

col1_file.write('\n'.join(col1))
col2_file.write('\n'.join(col2))