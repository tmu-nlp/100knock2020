import re

f = open('britain.txt', 'r')
britain = f.readline()

for line in britain.split("\\n"):
  if re.match(r"={2,4}", line):
    if re.match(r"={4}", line):
      print("3"+" "+line[4:-4])
    elif re.match(r"={3}", line):
      print("2"+" "+line[3:-3])
    else:
      print("1"+" "+line[2:-2])

f.close()
