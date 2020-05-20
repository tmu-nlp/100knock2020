import sys

file = open(sys.argv[1], 'r')
N = int(sys.argv[2])

for _ in range(N):
    print(file.readline(), end='')

