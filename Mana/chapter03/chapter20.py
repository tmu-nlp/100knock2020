import gzip
import json

jawiki_1 = gzip.open('jawiki-country.json.gz', 'rt')

for line in jawiki_1:
  jawiki_json = json.loads(line)
  if jawiki_json["title"]=="イギリス":
    britain = json.dumps(jawiki_json["text"], ensure_ascii=False)

jawiki_1.close()

with open('britain.txt', 'w') as f:
    f.write(britain)