# Code
from knock10 import read_file

lines = read_file()
for line in lines:
    line = line.replace('\t', ' ')
    print(line)

# Unix command
# $cat popular-names.txt | tr '\t' ' '
# $expand -t 1 popular-names.txt 
