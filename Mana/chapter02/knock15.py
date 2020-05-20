import sys


filename = open(sys.argv[1], 'r')

n = int(input())
for line in filename.readlines()[-n:]:
    print(line[:-1]) #remove '\n'

# tail -n [] 'merge.txt'