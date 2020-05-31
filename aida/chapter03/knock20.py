import gzip
import json

def read_gzip(file_path='jawiki-country.json.gz'):
    with gzip.open(file_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj[u"title"] == u"イギリス":
                return obj["text"]
    return None


if __name__ == '__main__':
    uk_doc = read_gzip()
    print(uk_doc)
