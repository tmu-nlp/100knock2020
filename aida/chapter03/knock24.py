import re
from knock20 import read_gzip

uk_doc = read_gzip()
figs = re.findall(r'\[\[(ファイル:)([^\||\]]+)[\||\]]', uk_doc)

for fig in figs:
    print(fig[1])

