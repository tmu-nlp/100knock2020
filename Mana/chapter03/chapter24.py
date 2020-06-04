import re

f = open('britain.txt', 'r')
britain = f.readline()

for line in britain.split("\\n"):
    if re.search(r"\[\[ファイル:(.*?)\|", line):
        print(re.search(r"\[\[ファイル:(.*?)\|", line).group(1))