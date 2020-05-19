import sys

with open(sys.argv[1], 'r') as file:
    col1 = []
    col2 = []

    for line in file:
        cols = line.split('\t')
        col1.append(cols[0])
        col2.append(cols[1])

    with open('col1.txt', 'w') as col1_file:
        col1_file.write('\n'.join(col1))

    with open('col2.txt', 'w') as col2_file:
        col2_file.write('\n'.join(col2))