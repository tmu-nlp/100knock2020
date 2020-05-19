import sys

with open(sys.argv[1], 'r') as file:
    col1 = [line.split('\t')[0] for line in file]
    counter = {}
    for item in col1:
        if item not in counter:
            counter[item] = 1
        else:
            counter[item] += 1
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print('\n'.join([item[0] + '\t' + str(item[1]) for item in counter]))
