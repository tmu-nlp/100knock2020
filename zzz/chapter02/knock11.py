import sys

file = open(sys.argv[1], 'r')

for line in file:
    new_line = line.replace('\t', ' ')
    print(line)
    print(new_line)