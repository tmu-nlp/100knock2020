import sys

with open(str(sys.argv[1])) as f:
    for line,i in zip(f,range(int(sys.argv[2]))):
        print(line)   