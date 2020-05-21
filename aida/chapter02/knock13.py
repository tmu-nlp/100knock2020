# Code
from knock10 import read_file

col1_path = './col1.txt'
col2_path = './col2.txt'

col1_lines = read_file(col1_path)
col2_lines = read_file(col2_path)

with open('./combine.txt', 'w') as fp:
    for col1, col2 in zip(col1_lines, col2_lines):
        fp.write(f'{col1}\t{col2}\n')


# Unix command
# $paste -d '\t' col1_unix col2_unix > combine
