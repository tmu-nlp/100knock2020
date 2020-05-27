file1 = "popular-names.txt"

with open(file1) as open_file:
    file_data = open_file.readlines()

print("{} {}".format(len(file_data), file1))

#wc -l popular-names.txt

#確認diff -s <(python3 knock10.py) <(wc -l popular-names.txt)
