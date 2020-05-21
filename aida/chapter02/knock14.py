# Code
from knock10 import read_file

lines = read_file()

print('please type number of lines')
print('n = ', end='')
n = int(input())

for line in lines[:n]:
    print(line)

# Unix command
# $head -n {number} popular-names.txt
