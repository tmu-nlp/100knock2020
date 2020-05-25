# Code
from knock10 import read_file

lines = read_file()

print('please type number of lines')
print('n = ', end='')
n = int(input())

for line in lines[-1-n:]:
    print(line)

# Unix command
# $tail -n {number} popular-names.txt
