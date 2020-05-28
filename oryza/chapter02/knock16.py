import sys
from more_itertools import chunked

with open(str(sys.argv[1])) as f:
    lines = f.readlines()
    num_f = int(len(lines) / int(sys.argv[2])) 
    splits = chunked(lines,num_f)

    for i,j in enumerate(splits):
        with open('split_file_' + str(i+1) + '.txt', 'w') as f_split:
            for line in j:
                f_split.writelines(line)