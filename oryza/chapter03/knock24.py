import sys
import re

file = open('united_kingdom.txt','r')

for line in file:
    compiler = re.compile(r'\[\[File:(.*?)(\||\])')
    if compiler.search(line):
        print(compiler.search(line).group(1))