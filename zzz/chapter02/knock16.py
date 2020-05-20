import sys

file = open(sys.argv[1], 'r')
N = int(sys.argv[2])
if len(sys.argv) > 3:
    output_file_num = sys.argv[3] + '{}.txt'
else:
    output_file_name = sys.argv[1].replace('.', '{}.')

cache = []
output_file_num = 0
for (index, line) in enumerate(file):
    cache.append(line.replace('\n', ''))
    if (index + 1) % N == 0:
        output_file = open(output_file_name.format(output_file_num), 'w')
        output_file.write('\n'.join(cache))
        output_file_num += 1
        cache = []
else:
    output_file = open(output_file_name.format(output_file_num), 'w')
    output_file.write('\n'.join(cache))

