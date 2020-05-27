from sys import argv

#N = int(input())

file1 = "popular-names.txt"

with open(file1) as open_file1:
    file1_data = open_file1.read()

file1_data = file1_data.splitlines()

for line in range(int(argv[1])):
    print(file1_data[line])

#diff -s <(python3 knock14.py 5) <(head -n 5 popular-names.txt)
