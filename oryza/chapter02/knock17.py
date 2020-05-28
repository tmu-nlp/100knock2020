import sys

unique_strings = []
with open(str(sys.argv[1])) as f:
    for i in f:
        line = i.split()[0]
        if line not in unique_strings:
            unique_strings.append(line)
            print(line)   