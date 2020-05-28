import sys

# tail -n N popular-names.txt

with open(sys.argv[1], 'r') as file:
    N = int(sys.argv[2])

    cache = [line.replace('\n', '') for line in file]
    print('\n'.join(cache[len(cache) - N: len(cache)]))

    # size = file.seek(0, 2)
    # file.seek(size - N * 50, 0)
    # cache = file.readlines()
    # print(''.join(cache[-N:]))
