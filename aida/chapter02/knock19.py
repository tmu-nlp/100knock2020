# Code
from knock10 import read_file

lines = read_file()

name_freq = {}
for line in lines:
    words = line.split('\t')
    name = words[0]
    if name in name_freq:
        name_freq[name] += 1
    else:
        name_freq[name] = 1

sorted_freq = sorted(name_freq.items(), key=lambda x:x[1], reverse=True)

for n_f in sorted_freq:
    name = n_f[0]
    freq = n_f[1]
    print(f'{name}\t{freq}')
    

# Unix command
# $cut -f 1 popular-names.txt | sort | uniq -c | sort -r
