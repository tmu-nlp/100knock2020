import sys

# paste col1.txt col2.txt

with open(sys.argv[1], 'r') as col1_file:
    with open(sys.argv[2], 'r') as col2_file:
        col1_2 = []

        for col1, col2 in zip(col1_file, col2_file):
            col1_2.append(col1.replace('\n', '') + '\t' + col2.replace('\n', ''))

        with open('col1_2.txt', 'w') as col1_2_file:
            col1_2_file.write('\n'.join(col1_2))
