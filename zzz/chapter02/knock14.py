import sys

with open(sys.argv[1], 'r') as file:
    N = int(sys.argv[2])

    for _ in range(N):
        print(file.readline(), end='')

