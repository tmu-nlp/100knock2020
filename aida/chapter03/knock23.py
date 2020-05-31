import re
from knock20 import read_gzip

uk_doc = read_gzip()
sections = re.findall(r'==+[^=]+==+', uk_doc)

for section in sections:
    level = (section.count('=') // 2) - 1
    section_name = re.sub(r'=|\s', '', section)
    print(f'{"#"*(level)} {section_name}\t{level}')

