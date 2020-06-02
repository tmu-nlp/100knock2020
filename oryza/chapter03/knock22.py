import sys
import re

file = open('united_kingdom.txt','r')

for line in file:
    compiler = re.compile(r'\[\[Category:(.*?)\]\]')
    if compiler.search(line):
        print(compiler.search(line).group(1))

# group() output the matched object