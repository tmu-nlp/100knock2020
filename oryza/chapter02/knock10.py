import sys

with open(str(sys.argv[1])) as f:
    read_data = f.read()
    data = read_data.split('\n')

print(len(data))
