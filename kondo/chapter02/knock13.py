import pathlib

file_data3 = pathlib.Path('./marged.txt')

file1 = "col1.txt"
file2 = "col2.txt"
file3 = "marged.txt"

with open(file1) as open_file1:
    file_data1 = open_file1.read()

with open(file2) as open_file2:
    file_data2 = open_file2.read()

file_data1 = file_data1.splitlines()
file_data2 = file_data2.splitlines()

for (name, gender) in zip(file_data1, file_data2):
    print(name + '\t' + gender)

#paste col1.txt col2.txt
#diff -s <(python3 knock13.py) <(paste col1.txt col2.txt)
