import pathlib

file1 = "popular-names.txt"

with open(file1) as open_file:
    file_data1 = open_file.read()

file_data1 = file_data1.splitlines()

file_data2 = pathlib.Path('./col1.txt')
file_data3 = pathlib.Path('./col2.txt')


with file_data2.open(mode='w') as open_file2:
    with file_data3.open(mode='w') as open_file3:
        for line in file_data1:
            line_list = line.split()
            open_file2.write(line_list[0] + '\n')
            open_file3.write(line_list[1] + '\n')

#cut --fields 1 popular-names.txt
#cut --fields 2 popular-names.txt

#diff -s <(cat col1.txt) <(cut --fields 1 popular-names.txt)
#diff -s <(cat col2.txt) <(cut --fields 2 popular-names.txt)
