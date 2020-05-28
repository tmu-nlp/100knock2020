from sys import argv

#N = int(input())

file = "popular-names.txt"

with open(file) as open_file:
    file_data = open_file.read()

file_data = file_data.splitlines()

for line in range(len(file_data) - int(argv[1]), len(file_data)):
    print(file_data[line])

#diff -s <(python3 knock15.py 5) <(tail -n 5 popular-names.txt)
