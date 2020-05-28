import sys

with open(str(sys.argv[1])) as col1, open(str(sys.argv[2])) as col2:
    with open(str(sys.argv[3]), 'w') as merge:
        for line1,line2 in zip(col1,col2):
            merge.write(line1.strip() + '\t' + line2.strip() + '\n')