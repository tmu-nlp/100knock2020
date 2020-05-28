import sys

with open(str(sys.argv[1])) as f:
    with open(str(sys.argv[2]), 'w') as col1, open(str(sys.argv[3]), 'w') as col2:
        for line in f:
            col1.write(line.split()[0] + '\n')
            col2.write(line.split()[1] + '\n')