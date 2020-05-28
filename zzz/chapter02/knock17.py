import sys

# cut -f1 popular-names.txt|sort|uniq

with open(sys.argv[1], 'r') as file:
    col1 = [line.split('\t')[0] for line in file]
    col1 = set(col1)
    col1 = sorted(col1)
    print('\n'.join(col1))
