import math

N = int(input('natural number : '))
with open('popular-names.txt') as file:
    #readlines()はファイルの内容を1行ずつ読み込み、読み込んだ内容をリストに格納します。
    lines = file.readlines()
    num_of_lines = len(lines)

start = 0
n = math.ceil(num_of_lines / N)
for i in range(N): #ceil():切り上げ
    end = start + n
    lis = lines[start:end]

    with open('f_' + str(i+1), 'w') as file:
        for line in lis:
           file.write(line)

    start += n
#ファイルを作成する
#split -l 4 popular-names.txt f_
