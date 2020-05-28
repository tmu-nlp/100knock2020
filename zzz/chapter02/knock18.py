import sys

# cut -f3 popular-names.txt|sort -n

with open(sys.argv[1], 'r') as file:
    cache = [line.replace('\n', '').split('\t') for line in file]
    sorted_cache = sorted(cache, key=lambda s: int(s[2]), reverse=True)
    res = ['\t'.join(line) for line in sorted_cache]
    print('\n'.join(res))
