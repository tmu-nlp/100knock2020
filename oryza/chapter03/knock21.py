import sys
import re

file = open('united_kingdom.txt','r')

for line in file:
    compiler = re.compile(r'\[\[Category:.*?\]\]')
    if compiler.search(line):
        print(line)


# [[Category:United Kingdom| ]]
# [[Category:British Islands]]
# [[Category:Countries in Europe]]
# [[Category:English-speaking countries and territories]]
# [[Category:G7 nations]]
# [[Category:Group of Eight nations]]
# [[Category:G20 nations]]
# [[Category:Island countries]]
# [[Category:Northern European countries]]
# [[Category:Former member states of the European Union]]
# [[Category:Member states of NATO]]
# [[Category:Member states of the Commonwealth of Nations]]
# [[Category:Member states of the Council of Europe]]
# [[Category:Member states of the Union for the Mediterranean]]
# [[Category:Member states of the United Nations]]
# [[Category:Priority articles for attention after Brexit]]
# [[Category:Western European countries]]