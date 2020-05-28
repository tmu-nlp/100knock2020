file = "popular-names.txt"

#readline() ファイルを各行が1要素のリストとして読み込む

with open(file) as open_file:
    file_data = open_file.read()

file_data = file_data.splitlines()

file_data.sort(key = lambda line: int(line.split('\t')[2]), reverse=True)

for line in file_data:
    print(line)

#sort -r -n -k 3,3 popular-names.txt
