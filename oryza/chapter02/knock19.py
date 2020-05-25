import sys
import operator

counts = {}
my_file = open(sys.argv[1])

words = []
for line in my_file:
    col1 = line.split()[0]
    if col1 in counts:
        counts[col1] +=1
    else:
        counts[col1] = 1

for x, y in sorted(counts.items(), key=operator.itemgetter(1), reverse=True):
    print ('%s: %r' % (x, y))