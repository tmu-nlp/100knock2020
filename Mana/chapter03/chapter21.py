import re

f = open('britain.txt', 'r')
britain = f.readline()

for line in britain.split("\\n"):
  if re.match(r'\[\[Category:', line):
    print(line)

f.close()
