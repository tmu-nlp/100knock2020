import sys
import re

file = open('united_kingdom.txt','r')
for line in file:
    compiler = re.compile(r'\=\=\=?(.*?)\=\=')
    if compiler.search(line):
        print(str(line.count('=',0,3) - 1) + '\t' + compiler.search(line).group(1))