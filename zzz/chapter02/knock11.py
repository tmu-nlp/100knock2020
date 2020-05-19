import sys

with open(sys.argv[1], 'r') as file:

    for line in file:
        new_line = line.replace('\t', ' ')
        print(line)
        print(new_line)