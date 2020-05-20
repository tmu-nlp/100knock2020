names = set()

file_col1 = open('col1.txt', 'r')
col1_lines = [line.strip() for line in file_col1.readlines()]

for line in col1_lines:
    names.add(line)

print(names)

#-u: option for listing words only once per word
#cut -f 1 -d " " popular-names.txt | sort -u (-o [output file name] col1.txt)