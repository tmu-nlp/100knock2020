import re
from knock20 import read_gzip

uk_doc = read_gzip()
categories = re.findall(r"\[\[Category:([^\]]+)\]\]", uk_doc)

for category in categories:
    print(category)

