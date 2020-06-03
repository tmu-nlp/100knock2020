import json
import gzip
import sys

def extract_text(gz_file,title,text_out):
    with gzip.open(gz_file,'rb') as fgz:
        for line in fgz:
            json_data = json.loads(line)
            title = title.replace('-',' ')
            if json_data['title'] == title:
                with open(text_out,'w') as outf:
                    outf.write(json_data['text'])

if __name__ == '__main__':
    extract_text(sys.argv[1],sys.argv[2],sys.argv[3])

# python3 knock20.py enwiki-country.json.gz United-Kingdom united_kingdom.txt
# python3 knock20.py gzfile title(replace space with -) output_file