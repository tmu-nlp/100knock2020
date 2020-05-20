import sys

filename = open(sys.argv[1], 'r')

n = int(input())
for line in filename.readlines()[0:n]:
    print(line.strip())

#head -n [] merge.txt