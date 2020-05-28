import sys

with open(str(sys.argv[1])) as f:
    lines = f.readlines()
    lines.sort(key=lambda l: float(l.split()[2]),reverse=False)
    for line in lines:
        line.replace('\n','').split('\t')
        print(line)

