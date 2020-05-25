import sys

with open(str(sys.argv[1])) as f:
    for line in (f.readlines() [-int(sys.argv[2]):]):
        print(line,end='\n')   