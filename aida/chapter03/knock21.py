import re
from knock20 import read_gzip

uk_doc = read_gzip()
category_lines = re.findall(r"\[\[Category:[^\]]+\]\]", uk_doc)

for category_line in category_lines:
    print(category_line)

