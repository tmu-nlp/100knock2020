import sys

file = open(sys.argv[1], 'r')
line_counter = 0

for line in file:
    line_counter += 1

print('Number of lines:', line_counter)