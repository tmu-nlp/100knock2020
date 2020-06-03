import re

f = open('britain.txt', 'r')
britain = f.readline()

for line in britain.split("\\n"):
  if re.match(r'\[\[Category:', line):
    if re.search(r':(.*?)\|', line):
      print(re.search(r':(.*?)\|', line).group()[1:-1])
    else:
      print(re.search(r':(.*?)\]', line).group()[1:-1])

f.close()