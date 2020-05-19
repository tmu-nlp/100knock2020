import sys

with open(sys.argv[1], 'r') as file:
    N = int(sys.argv[2])

    cache = [line.replace('\n', '') for line in file]
    print('\n'.join(cache[len(cache) - N: len(cache)]))
