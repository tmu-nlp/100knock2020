import sys

# cat popular-names.txt | tr '\t' ' '

with open(sys.argv[1], 'r') as file:

    for line in file:
        new_line = line.replace('\t', ' ')
        print(new_line, end='')