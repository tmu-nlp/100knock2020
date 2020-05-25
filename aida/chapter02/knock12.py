# Code
from knock10 import read_file

lines = read_file()
col1 = []
col2 = []

for line in lines:
    words = line.split('\t')
    col1.append(words[0])
    col2.append(words[1])

with open('./col1.txt', 'w') as fp:
    for word in col1:
        fp.write(f'{word}\n')

with open('./col2.txt', 'w') as fp:
    for word in col2:
        fp.write(f'{word}\n')

# Unix command
# $cut -f 1 popular-names.txt > col1_unix
# $cut -f 2 popular-names.txt > col2_unix
