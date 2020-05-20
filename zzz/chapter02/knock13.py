import sys

col1_file = open(sys.argv[1], 'r')
col2_file = open(sys.argv[2], 'r')

col1_2 = []

for col1, col2 in zip(col1_file, col2_file):
    col1_2.append(col1.replace('\n', '') + '\t' + col2.replace('\n', ''))

col1_2_file = open('col1_2.txt', 'w')
col1_2_file.write('\n'.join(col1_2))